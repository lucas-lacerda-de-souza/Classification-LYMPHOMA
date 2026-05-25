import os
import csv
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import openslide

import torch
import torch.nn as nn
from torchvision import transforms
import timm


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------
parser = argparse.ArgumentParser(
    description="UNI + MIL inference for head and neck lymphoid lesion classification"
)

parser.add_argument("--input_dir", type=str, required=True,
                    help="Directory containing .svs whole-slide images")

parser.add_argument("--output_dir", type=str, required=True,
                    help="Directory to save inference results")

parser.add_argument("--uni_weights", type=str, required=True,
                    help="Path to UNI weights: pytorch_model.bin")

parser.add_argument("--patch_model_weights", type=str, required=True,
                    help="Path to trained patch classifier weights")

parser.add_argument("--mil_model_weights", type=str, required=True,
                    help="Path to trained MIL model weights")

parser.add_argument("--patch_size", type=int, default=224)
parser.add_argument("--stride", type=int, default=448)
parser.add_argument("--max_patches", type=int, default=None)

args = parser.parse_args()


# ---------------------------------------------------------
# Classes
# ---------------------------------------------------------
CLASS_NAMES = {
    0: "Aggressive B-cell lymphoma",
    1: "Indolent/Small B-cell lymphoma",
    2: "NK/T-cell lymphoma",
    3: "Reactive lesion"
}

NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ---------------------------------------------------------
# UNI Feature Extractor
# ---------------------------------------------------------
def load_uni_model(weights_path):
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True
    )

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    model = model.to(DEVICE)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


# ---------------------------------------------------------
# Patch Classifier
# ---------------------------------------------------------
class PatchClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# MIL Attention Model
# ---------------------------------------------------------
class MILAttention(nn.Module):
    def __init__(self, embedding_dim, num_classes=4):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_attention=False):
        attention_scores = self.attention(x)
        attention_weights = torch.softmax(attention_scores, dim=0)

        slide_embedding = torch.sum(attention_weights * x, dim=0)
        logits = self.classifier(slide_embedding)

        if return_attention:
            return logits, attention_weights

        return logits


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def remove_module_prefix(state_dict):
    return {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }


def is_tissue_patch(patch, white_threshold=220, tissue_fraction_threshold=0.20):
    arr = np.array(patch)
    gray = arr.mean(axis=2)
    tissue_fraction = np.mean(gray < white_threshold)
    return tissue_fraction > tissue_fraction_threshold


def discover_svs(input_dir):
    svs_files = sorted(Path(input_dir).rglob("*.svs"))
    return [str(p) for p in svs_files]


# ---------------------------------------------------------
# Extract UNI embeddings from one WSI
# ---------------------------------------------------------
def extract_slide_embeddings(slide_path, uni_model):
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.dimensions

    xs = list(range(0, max(1, width - args.patch_size + 1), args.stride))
    ys = list(range(0, max(1, height - args.patch_size + 1), args.stride))

    coords = [(x, y) for y in ys for x in xs]

    if args.max_patches is not None:
        coords = coords[:args.max_patches]

    embeddings = []
    valid_coords = []

    with torch.no_grad():
        for x, y in tqdm(coords, desc=f"Processing {Path(slide_path).name}", leave=False):
            patch = slide.read_region(
                (x, y),
                0,
                (args.patch_size, args.patch_size)
            ).convert("RGB")

            if not is_tissue_patch(patch):
                continue

            patch_tensor = transform(patch).unsqueeze(0).to(DEVICE)

            embedding = uni_model(patch_tensor)

            if isinstance(embedding, (list, tuple)):
                embedding = embedding[0]

            embeddings.append(embedding.squeeze(0).cpu())
            valid_coords.append((x, y))

    slide.close()

    if len(embeddings) == 0:
        return None, []

    embeddings = torch.stack(embeddings, dim=0)

    return embeddings, valid_coords


# ---------------------------------------------------------
# Inference for one slide
# ---------------------------------------------------------
def predict_slide(slide_path, uni_model, patch_model, mil_model):
    embeddings, coords = extract_slide_embeddings(slide_path, uni_model)

    if embeddings is None:
        return None

    embeddings = embeddings.to(DEVICE)

    with torch.no_grad():
        # Patch-level prediction
        patch_logits = patch_model(embeddings)
        patch_probs = torch.softmax(patch_logits, dim=1)

        mean_patch_probs = patch_probs.mean(dim=0)

        # MIL slide-level prediction
        mil_logits, attention_weights = mil_model(
            embeddings,
            return_attention=True
        )

        mil_probs = torch.softmax(mil_logits, dim=0)

        predicted_class = torch.argmax(mil_probs).item()
        confidence = mil_probs[predicted_class].item()

    result = {
        "slide": Path(slide_path).name,
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "prob_aggressive_b_cell": mil_probs[0].item(),
        "prob_indolent_small_b_cell": mil_probs[1].item(),
        "prob_nk_t_cell": mil_probs[2].item(),
        "prob_reactive": mil_probs[3].item(),
        "mean_patch_prob_aggressive_b_cell": mean_patch_probs[0].item(),
        "mean_patch_prob_indolent_small_b_cell": mean_patch_probs[1].item(),
        "mean_patch_prob_nk_t_cell": mean_patch_probs[2].item(),
        "mean_patch_prob_reactive": mean_patch_probs[3].item(),
        "n_valid_patches": len(coords)
    }

    return result


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading UNI model...")
    uni_model = load_uni_model(args.uni_weights)

    print("Preparing classifier dimensions...")
    dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)

    with torch.no_grad():
        dummy_embedding = uni_model(dummy)

        if isinstance(dummy_embedding, (list, tuple)):
            dummy_embedding = dummy_embedding[0]

    embedding_dim = dummy_embedding.shape[1]

    print(f"UNI embedding dimension: {embedding_dim}")

    patch_model = PatchClassifier(
        embedding_dim=embedding_dim,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    mil_model = MILAttention(
        embedding_dim=embedding_dim,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    patch_state = torch.load(args.patch_model_weights, map_location=DEVICE)
    mil_state = torch.load(args.mil_model_weights, map_location=DEVICE)

    patch_model.load_state_dict(remove_module_prefix(patch_state), strict=True)
    mil_model.load_state_dict(remove_module_prefix(mil_state), strict=True)

    patch_model.eval()
    mil_model.eval()

    svs_files = discover_svs(args.input_dir)

    if len(svs_files) == 0:
        raise RuntimeError("No .svs files found in input directory.")

    results = []

    for slide_path in svs_files:
        result = predict_slide(
            slide_path=slide_path,
            uni_model=uni_model,
            patch_model=patch_model,
            mil_model=mil_model
        )

        if result is not None:
            results.append(result)
            print(
                f"{result['slide']}: "
                f"{result['predicted_label']} "
                f"(confidence={result['confidence']:.4f})"
            )
        else:
            print(f"{Path(slide_path).name}: no valid tissue patches found")

    output_csv = os.path.join(args.output_dir, "inference_results.csv")

    fieldnames = [
        "slide",
        "predicted_class",
        "predicted_label",
        "confidence",
        "prob_aggressive_b_cell",
        "prob_indolent_small_b_cell",
        "prob_nk_t_cell",
        "prob_reactive",
        "mean_patch_prob_aggressive_b_cell",
        "mean_patch_prob_indolent_small_b_cell",
        "mean_patch_prob_nk_t_cell",
        "mean_patch_prob_reactive",
        "n_valid_patches"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow(r)

    print("\n✓ Inference completed.")
    print(f"Results saved to: {output_csv}")


if __name__ == "__main__":
    main()
