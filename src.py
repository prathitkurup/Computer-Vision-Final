'''
CSCI 3485
Final Project: Detecting Real vs. AI-Generated Images
Victoria Figueroa & Prathit Kurup

Outline:
1) Download the HF dataset and load it in
2) Create a PyTorch dataloader for the dataset
3) Create train and test sets that mix both real and AI-generated images (Label_A = 0 means real)
4) Build a binary classifier using transfer learning with ResNet-50
    4a) Train the model and evaluate its performance on the test set
    4b) Binary Cross Entropy Loss and ADAM optimizer with a learning rate 1e-3
5) Test the model on the test set to compare performance
    5a) Quantitative: compare the classification accuracy of both models
    5b) Qualitatively: visually compare the images
'''

from datasets import load_dataset
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch.nn as nn
from torchvision.io import read_image
from torchsummary import summary
from torch.optim import Adam
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

LOSS_FN = nn.BCEWithLogitsLoss()
LR = 1e-3
NUM_CLASSES = 2
BATCH_SIZE = 128
N_EPOCHS = 10
MLP_NEURONS = 512

class HFImageDataset(Dataset):
    '''Custom PyTorch Dataset to wrap the Hugging Face dataset split.'''
    def __init__(self, hf_split, transform=None):
        '''hf_split: a Hugging Face dataset split (e.g. ds["train"])'''
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        '''Return the number of samples in the dataset.'''
        return len(self.data)

    def __getitem__(self, idx):
        '''Return the image and label for a given index.'''
        item = self.data[idx]
        img = item["Image"].convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = item["Label_A"]   # 0 = Real, 1 = AI-Generated
        return img, label

def create_data_loader(transform=None):
    '''Create PyTorch DataLoaders for the training and test sets.'''
    ds = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset")
    train_dataset = HFImageDataset(ds["train"], transform=transform)
    test_dataset  = HFImageDataset(ds["test"],  transform=transform)
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
    return train_dl, test_dl

def build_mlp_model(input_dim, num_classes=10):
    '''Build a simple MLP model to replace the original classifier of the pre-trained model.'''
    model = nn.Sequential(
        nn.Linear(input_dim, MLP_NEURONS),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(MLP_NEURONS, num_classes)).to(device)
    # summary(model, (input_dim,), device="cpu")
    return model

def train_batch(x, y, model, opt, loss_fn):
    '''Train the model on a single batch of data and return the loss and predictions.'''
    model.train()
    opt.zero_grad()
    outputs = model(x).squeeze(1)
    batch_loss = loss_fn(outputs, y.float())
    batch_loss.backward()
    opt.step()
    preds = (outputs > 0).long()

    return batch_loss.detach().cpu(), preds.detach()

def train_model(model, model_name, train_dl):
    '''Train the model for N_EPOCHS and return the list of losses and accuracies at each epoch.'''
    loss_fn = LOSS_FN
    opt = Adam(model.parameters(), lr=LR)

    losses, accuracies = [], []
    start_time = time.time()
    print(f"\nBegin training for: {model_name}")
    for epoch in range(N_EPOCHS):
        print(f"{model_name}: epoch {epoch + 1} of {N_EPOCHS}")
        epoch_losses = []
        correct = 0
        total = 0

        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            batch_loss, preds = train_batch(x, y, model, opt, loss_fn)
            epoch_losses.append(batch_loss.item())
            correct += (preds == y).sum().item()
            total += y.size(0)

        epoch_loss = float(np.mean(epoch_losses))
        epoch_accuracy = correct / total

        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        print(f"  loss={epoch_loss:.4f}, train_acc={epoch_accuracy:.4f}")

    end_time = time.time()
    train_time = end_time - start_time
    print(f"{model_name} training time (seconds): {train_time}")
    return losses, accuracies, train_time

@torch.no_grad()
def accuracy(x, y, model):
    '''Evaluate the model on a batch of data and return the accuracy.'''
    model.eval()
    prediction = model(x).squeeze(1)
    preds = (prediction > 0).long()
    s = torch.sum((preds == y).float()) / len(y)
    return s.cpu().numpy()

@torch.no_grad()
def test_model(model, model_name,test_dl):
    '''Evaluate the model on the test set and print the average accuracy.'''
    accs = []
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        accs.append(accuracy(x, y, model))
    print(f"{model_name} test accuracy: {np.mean(accs)}")

def visualize_training(losses, accuracies, model_name):
    '''Visualize the training loss and accuracy curves.'''
    print(f"\nVisualizing training for {model_name}\n")
    plt.figure(figsize=(13,3))
    plt.subplot(121)
    plt.title(f'{model_name}: Training Loss')
    plt.plot(np.arange(N_EPOCHS) + 1, losses)
    plt.subplot(122)
    plt.title(f'{model_name}: Training Accuracy')
    plt.plot(np.arange(N_EPOCHS) + 1, accuracies)
    plt.show()

def freeze_backbone(model):
    '''Freeze the backbone of the pre-trained model so its weights are not updated during training.'''
    model.to(device)
    # model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

@torch.no_grad()
def run_conv_layers(dl, model):
    '''Run the convolutional layers of the pre-trained model on the dataset and return the extracted features and labels.'''
    print("Extracting features using frozen backbone...")

    start_time = time.time()
    model.eval()
    # Run convolutional layers once to extract features using the pre-trained weights
    features = []
    labels = []
    for x, y in dl:
        x = x.to(device)
        outputs = model(x)      # run CNN forward once
        outputs = torch.flatten(outputs, start_dim=1)
        features.append(outputs.detach().cpu())
        labels.append(y)

    end_time = time.time()
    extraction_time = end_time - start_time
    print("CNN feature extraction time (seconds): ", extraction_time)

    # This is the data we will train the MLP on
    return torch.cat(features), torch.cat(labels)

# def extract_resnet_features(resnet, x):
#     feats = resnet.conv1(x)
#     feats = resnet.bn1(feats)
#     feats = resnet.relu(feats)
#     feats = resnet.maxpool(feats)
#     feats = resnet.layer1(feats)
#     feats = resnet.layer2(feats)
#     feats = resnet.layer3(feats)
#     feats = resnet.layer4(feats)
#     feats = resnet.avgpool(feats)
#     feats = torch.flatten(feats, 1) # [B, C*H*W]
#     return feats

def run_experiment(model, model_weights, model_name):
    '''Run the full training and evaluation pipeline for a given pre-trained model and return the trained backbone and MLP classifier.'''
    print(f"\nRunning experiment for {model_name}")

    # Dataset using the correct preprocessing
    model_transformations = model_weights.transforms()
    train_dl, test_dl = create_data_loader(model_transformations)

    # Remove classifier from backbone for ResNet50 feature extraction
    model.fc = nn.Identity()

    backbone = model.to(device)

    # Freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    backbone.eval()
    print(f"\n{model_name} backbone summary:")
    summary(backbone, (3, 224, 224), device=device)

    train_features, train_labels = run_conv_layers(train_dl, backbone)
    test_features, test_labels = run_conv_layers(test_dl, backbone)

    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build MLP classifier
    input_dim = train_features.shape[1]
    mlp = build_mlp_model(input_dim=input_dim, num_classes=1).to(device)
    print(f"\n{model_name} MLP classifier summary:")
    summary(mlp, (input_dim,), device=device)

    # Train classifier
    losses, accuracies, train_time = train_model(
        model=mlp,
        model_name=model_name,
        train_dl=train_dl
    )

    # Visualization
    visualize_training(losses, accuracies, model_name)
    print(f"Training time for {model_name}: {train_time:.2f} seconds")

    # Test accuracy
    test_model(mlp, model_name, test_dl)

    backbone_params = sum(p.numel() for p in backbone.parameters())
    mlp_params = sum(p.numel() for p in mlp.parameters())
    print(f"Total parameters in {model_name} backbone: {backbone_params}")
    print(f"Total parameters in {model_name} MLP: {mlp_params}")

    return backbone, mlp

def get_gradcam(mlp, backbone, img_tensor, genre_tensor):
    """
    Returns a 2-D numpy array [7, 7] with values in [0, 1].
    High values = regions the model weighted most heavily.
    """
    target_layer = backbone.layer4   # last conv block of ResNet50

    activations = [None]
    gradients   = [None]

    def fwd_hook(_, __, output):
        activations[0] = output

    def bwd_hook(_, __, grad_out):
        gradients[0] = grad_out[0]

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    # Keep backbone in eval mode (preserves BN running stats) but
    # temporarily enable gradients so backprop can reach layer4.
    for p in backbone.parameters():
        p.requires_grad_(True)

    img  = img_tensor.unsqueeze(0).to(device)
    gen  = genre_tensor.unsqueeze(0).to(device)

    with torch.enable_grad():
        feat_flat = torch.flatten(backbone(img), 1)          # [1, 2048]
        pred      = mlp(torch.cat([feat_flat, gen], dim=1)).squeeze()
        backbone.zero_grad()
        mlp.zero_grad()
        pred.backward()

    h_fwd.remove()
    h_bwd.remove()

    # Re-freeze backbone
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Grad-CAM formula: global-average-pool the gradients over spatial dims,
    # use them as channel weights, then ReLU the weighted sum of activations.
    grads   = gradients[0]                              # [1, 2048, 7, 7]
    acts    = activations[0]                            # [1, 2048, 7, 7]
    weights = grads.mean(dim=[2, 3], keepdim=True)      # [1, 2048, 1, 1]
    cam     = F.relu((weights * acts).sum(dim=1))       # [1,   7, 7]
    cam     = cam.squeeze().detach().cpu().numpy()

    # Normalise to [0, 1]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def show_gradcam_grid(mlp, backbone, dataset, indices=None, n=19,alpha_heatmap=0.45, colormap="jet"):
    """
    Plots n images each shown as:
      LEFT  — raw image(AI or Real)  +  true class label vs. predicted class label
      RIGHT — Grad-CAM overlay (hotter = more attended)
    Parameters:
    indices   : list of ints — specific dataset indices to show.
                If None, picks n at random.
    alpha_heatmap : blend strength of the heatmap (0 = invisible, 1 = full)
    colormap  : matplotlib colormap name ('jet', 'hot', 'plasma', etc.)
    """
    # Grab raw image from hugging face dataset, not drive, also need to change output to show classification results instead of regression results
    # Need to update this for our binary classification task — show predicted vs. true label instead of gross, and change the title accordingly
    dataset = create_data_loaders(True)
    if indices is None:
        indices = np.random.choice(len(dataset), size=n, replace=False).tolist()
    else:
        n = len(indices)

    cmap = plt.get_cmap(colormap)
    fig, axes = plt.subplots(n, 2, figsize=(9, 4.5 * n))
    if n == 1:
        axes = [axes]

    mlp.eval()
    backbone.eval()

    for row, idx in enumerate(indices):
        img_tensor, class_label, _ = dataset[idx]

        raw_pil = Image.open(dataset.paths[idx]).convert("RGB")
        img_np  = np.array(raw_pil) / 255.0

        # Grad-CAM
        cam = get_gradcam(mlp, backbone, img_tensor, genre_tensor)

        # Upsample 7×7 --> image resolution
        cam_up = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                raw_pil.size, resample=Image.BILINEAR
            )
        ) / 255.0

        heatmap = cmap(cam_up)[..., :3]                  # RGB, ignore alpha
        overlay = np.clip((1 - alpha_heatmap) * img_np + alpha_heatmap * heatmap, 0, 1)

        # Predicted vs. true class
        with torch.no_grad():
            feat_flat = torch.flatten(backbone(img_tensor.unsqueeze(0).to(device)), 1)
            pred_log  = mlp(
                torch.cat([feat_flat,
                           genre_tensor.unsqueeze(0).to(device)], dim=1)
            ).item()

        pred_class = pred_log
        true_class = label.item()

        # Plot
        axes[row][0].imshow(img_np)
        axes[row][0].set_title(
            f"Sample #{idx}\n"
            f"True:  {true_class}\n"
            f"Pred:  {pred_class}",
            fontsize=8, family="monospace", loc="left"
        )
        axes[row][0].axis("off")

        axes[row][1].imshow(overlay)

        # Colorbar on the right panel
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(0, 1))
        plt.colorbar(sm, ax=axes[row][1], fraction=0.03, pad=0.02,
                     label="Attention intensity")
        axes[row][1].set_title("Grad-CAM attention map", fontsize=9)
        axes[row][1].axis("off")

    plt.suptitle("ResNet50 Grad-CAM  \u00b7  Real vs. AI generated images detection", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()

def main():
    # Transfer learning with ResNet-50
    resnet50_weights = models.ResNet50_Weights.IMAGENET1K_V1
    resnet50_model = models.resnet50(weights=resnet50_weights)
    resnet50_backbone, resnet50_mlp = run_experiment(
        model=resnet50_model,
        model_weights=resnet50_weights,
        model_name="ResNet50"
    )
    

    #Grad Cam
    # Mode 1 — 4 random test posters
    # show_gradcam_grid(mlp, resnet_backbone, test_ds, n=4)

    # Mode 2 — specific poster indices you choose
    # show_gradcam_grid(mlp, resnet_backbone, test_ds, indices=[0, 12, 47, 99])

    # Mode 3 — top-5 highest predicted grossing posters in the test set
    # with torch.no_grad():
    #     all_preds = []
    #     for img_t, gen_t, _ in test_ds: # Removed vote_t
    #         feat = torch.flatten(resnet_backbone(img_t.unsqueeze(0).to(device)), 1)
    #         p = mlp(torch.cat([feat, gen_t.unsqueeze(0).to(device)], dim=1)).item() # Removed vote_t
    #         all_preds.append(p)
    # top20 = np.argsort(all_preds)[-5:][::-1].tolist()
    # show_gradcam_grid(mlp, resnet_backbone, test_ds, indices=top20)

def __innit__():
    main()