"""
visualize_gradcam.py — GradCAM heatmap comparison for FL unlearning.

For each requested case, loads the saved global model and generates GradCAM
overlays on clean and backdoored test images, saving two comparison grids:
  results/gradcam/gradcam_clean_{dataset}.png
  results/gradcam/gradcam_backdoor_{dataset}.png

Columns: [Original] + [one column per case]
Rows:    one row per sample image

Usage:
    python visualize_gradcam.py --dataset cifar10 --cases 0,1,2,3 --n_samples 4
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils.dataloader import get_loaders
from utils.model import BasicBlock, ResNet, get_model


CASE_LABELS = {
    0: "Case 0\n(Baseline)",
    1: "Case 1\n(Retrain)",
    2: "Case 2\n(Continue)",
    3: "Case 3\n(PGA)",
    4: "Case 4\n(FedEraser)",
    5: "Case 5\n(Flipping)",
}


def get_args():
    parser = argparse.ArgumentParser(description="GradCAM visualization for FL unlearning")

    # Inherited config.py parameters — used to reconstruct model filenames
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_rounds", type=int, default=20)
    parser.add_argument("--num_unlearn_rounds", type=int, default=5)
    parser.add_argument("--num_post_training_rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--poisoned_percent", type=float, default=0.9)

    # Visualization-specific parameters
    parser.add_argument(
        "--cases",
        type=str,
        default="0,1,2,3,4,5",
        help="Comma-separated case numbers to compare (e.g. '0,1,3')",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=-1,
        help="Which round's model to use (-1 = last available round per case)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of sample images to show per image type",
    )

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.loss_fn = torch.nn.CrossEntropyLoss()
    args.case_list = [int(c.strip()) for c in args.cases.split(",")]
    return args


def build_filename_stem(args, case):
    """Reconstruct the model filename stem (no _round{r}.pt suffix)."""
    return (
        f"case{case}_"
        f"{args.dataset}_"
        f"C{args.num_clients}_"
        f"BS{args.batch_size}_"
        f"R{args.num_rounds}_"
        f"UR{args.num_unlearn_rounds}_"
        f"PR{args.num_post_training_rounds}_"
        f"E{args.local_epochs}_"
        f"LR{args.lr}"
    )


def find_model_path(args, case, target_round=-1):
    """
    Return (Path, round_number) for the requested case.

    If target_round == -1, picks the highest round number found.
    Returns (None, None) if no matching file exists.
    """
    model_dir = Path(f"./results/models/case{case}")
    if not model_dir.exists():
        print(f"[case {case}] model directory not found: {model_dir}")
        return None, None

    stem = build_filename_stem(args, case)
    pattern = re.compile(rf"^{re.escape(stem)}_round(\d+)\.pt$")

    found = {}
    for f in model_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            found[int(m.group(1))] = f

    if not found:
        print(f"[case {case}] no matching .pt files in {model_dir}")
        return None, None

    if target_round == -1:
        chosen_round = max(found.keys())
    elif target_round in found:
        chosen_round = target_round
    else:
        print(
            f"[case {case}] round {target_round} not found; "
            f"falling back to last round."
        )
        chosen_round = max(found.keys())

    return found[chosen_round], chosen_round


def load_model_with_weights(args, case):
    """
    Instantiate the model and load saved weights for the given case.

    Returns (model, round_used) or (None, None) on failure.
    """
    model_path, used_round = find_model_path(args, case, target_round=args.round)
    if model_path is None:
        return None, None

    print(f"[case {case}] loading {model_path.name} (round {used_round})")
    # get_model() hardcodes 10 classes; cifar100 needs 100 classes
    if args.dataset == "cifar100":
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100).to(args.device)
    else:
        model = get_model(args)
    state_dict = torch.load(model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, used_round


def get_target_layer(model, dataset):
    """Return the list of GradCAM target layers for the given model/dataset."""
    if dataset == "mnist":
        # FLNet: use the second conv layer
        return [model.conv2]
    else:
        # Custom ResNet18 (cifar10 / cifar100): last BasicBlock of conv5_x
        return [model.conv5_x[-1]]


def get_sample_images(loader, n_samples, device):
    """Pull n_samples images (and labels) from a DataLoader."""
    imgs_list, lbls_list = [], []
    total = 0
    for imgs, lbls in loader:
        imgs_list.append(imgs)
        lbls_list.append(lbls)
        total += imgs.shape[0]
        if total >= n_samples:
            break
    images = torch.cat(imgs_list, dim=0)[:n_samples].to(device)
    labels = torch.cat(lbls_list, dim=0)[:n_samples].to(device)
    return images, labels


def tensor_to_rgb(img_tensor):
    """
    Convert a [C, H, W] tensor with values in [0, 1] to a
    float32 [H, W, 3] numpy array for use as a GradCAM overlay base.
    """
    img = img_tensor.detach().cpu().numpy()
    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)  # grayscale → RGB
    img = np.transpose(img, (1, 2, 0))  # [C,H,W] → [H,W,C]
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def generate_gradcam_grid(args, cases, clean_loader, backdoor_loader):
    """
    Build and save two comparison figures (clean + backdoor).

    Layout:
        rows = n_samples sample images
        cols = [Original] + [one GradCAM overlay per case]
    """
    n_samples = args.n_samples
    n_cols = 1 + len(cases)

    # Collect sample images once; reuse across cases
    clean_imgs, _ = get_sample_images(clean_loader, n_samples, args.device)
    bd_imgs, _ = get_sample_images(backdoor_loader, n_samples, args.device)

    for img_set, set_name in [(clean_imgs, "clean"), (bd_imgs, "backdoor")]:
        fig, axes = plt.subplots(
            n_samples,
            n_cols,
            figsize=(3 * n_cols, 3 * n_samples),
            squeeze=False,
        )

        # --- Column 0: original images ---
        axes[0, 0].set_title("Original", fontsize=10, fontweight="bold")
        for row in range(n_samples):
            axes[row, 0].imshow(tensor_to_rgb(img_set[row]))
            axes[row, 0].axis("off")

        # --- Columns 1…: GradCAM per case ---
        for col_idx, case in enumerate(cases, start=1):
            axes[0, col_idx].set_title(
                CASE_LABELS.get(case, f"Case {case}"),
                fontsize=9,
                fontweight="bold",
            )

            model, _ = load_model_with_weights(args, case)
            if model is None:
                for row in range(n_samples):
                    axes[row, col_idx].text(
                        0.5, 0.5, "N/A", ha="center", va="center", transform=axes[row, col_idx].transAxes
                    )
                    axes[row, col_idx].axis("off")
                continue

            target_layers = get_target_layer(model, args.dataset)
            cam = GradCAM(model=model, target_layers=target_layers)

            for row in range(n_samples):
                img_tensor = img_set[row]  # [C, H, W]
                rgb_img = tensor_to_rgb(img_tensor)
                input_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]

                grayscale_cam = cam(input_tensor=input_tensor, targets=None)
                cam_image = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)

                axes[row, col_idx].imshow(cam_image)
                axes[row, col_idx].axis("off")

        plt.tight_layout()

        out_dir = Path("results/gradcam")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"gradcam_{set_name}_{args.dataset}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def main():
    args = get_args()
    print(
        f"Dataset: {args.dataset} | Cases: {args.case_list} | "
        f"Samples: {args.n_samples} | Device: {args.device}"
    )

    # get_loaders needs these attrs; set safe defaults for visualization
    args.is_onboarding = False
    args.saved = False

    _, test_loader, test_loader_poison = get_loaders(args)

    generate_gradcam_grid(args, args.case_list, test_loader, test_loader_poison)
    print("Done.")


if __name__ == "__main__":
    main()
