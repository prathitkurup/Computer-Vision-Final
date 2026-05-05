import torch
import clip
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print("Using device:", device)

def group_by_caption(hf_split):
    """Group dataset sample indices by their shared caption (6 images per group)."""
    groups = defaultdict(list)
    for idx in range(len(hf_split)):
        caption = hf_split[idx]["Caption"]
        groups[caption].append(idx)
    return groups

def analyze_clip_confidence(hf_split, caption_groups, num_samples=50):
    """
    For each caption group (1 real + 5 AI images), compute CLIP image-text
    similarity scores and report how often the real image ranks highest.
    """
    real_wins = 0
    total_groups = 0
    results = []

    for caption in list(caption_groups.keys())[:num_samples]:
        indices = caption_groups[caption]
        if len(indices) != 6:
            continue

        items = [hf_split[i] for i in indices]
        images = [item["Image"].convert("RGB") for item in items]
        labels = [item["Label_A"] for item in items]  # 0 = Real, 1 = AI-Generated

        image_inputs = torch.stack([preprocess(img) for img in images]).to(device)
        text_token = clip.tokenize([caption], truncate=True).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_inputs).float()
            text_features = model.encode_text(text_token).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(1).cpu().numpy()

        if 0 in labels:
            real_idx = labels.index(0)
            if similarity[real_idx] == similarity.max():
                real_wins += 1
        total_groups += 1
        results.append((caption, images, labels, similarity))

    print(f"Real image ranked highest: {real_wins}/{total_groups} groups "
          f"({100 * real_wins / total_groups:.1f}%)")
    return results


def visualize_results(results, num_display=3):
    """Display images and CLIP similarity scores for a few caption groups."""
    for caption, images, labels, similarity in results[:num_display]:
        n = len(images)
        fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
        fig.suptitle(caption[:100], fontsize=9)

        for i, (img, label, score) in enumerate(zip(images, labels, similarity)):
            axes[i].imshow(img)
            tag = "REAL" if label == 0 else "AI"
            color = "green" if label == 0 else "red"
            axes[i].set_title(f"{tag}\n{score:.3f}", fontsize=9, color=color)
            axes[i].axis("off")

        bar_colors = ["green" if l == 0 else "red" for l in labels]
        x_labels = [f"{'REAL' if l == 0 else 'AI'}{i + 1}" for i, l in enumerate(labels)]
        axes[-1].bar(x_labels, similarity, color=bar_colors)
        axes[-1].set_ylabel("CLIP similarity")
        axes[-1].set_title("Scores by image")
        plt.tight_layout()
        plt.show()


def main():
    ds = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset")
    hf_split = ds["train"]

    caption_groups = group_by_caption(hf_split)
    print(f"Total caption groups: {len(caption_groups)}")

    results = analyze_clip_confidence(hf_split, caption_groups, num_samples=50)
    visualize_results(results, num_display=3)


if __name__ == "__main__":
    main()
