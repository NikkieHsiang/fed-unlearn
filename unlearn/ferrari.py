"""
Ferrari: Federated Feature Unlearning via Optimizing Feature Sensitivity
Reference: https://arxiv.org/abs/2405.17462

Core idea:
    Minimize the feature sensitivity of the model on the unlearning client's
    data w.r.t. the backdoor trigger region F:

        L = (1/N) * Σ_i  ||f_θ(x) - f_θ(x + δ_{F,i})||₂ / ||δ_{F,i}||₂

    where  δ_{F,i} ~ N(0, σ²)  with non-trigger pixels zeroed out.

    Minimising L makes the model insensitive to trigger-region perturbations,
    effectively erasing the backdoor.

    An optional cross-entropy regularisation term can be added:

        L_total = L_ferrari + lambda_reg * L_CE(f_θ(x), y)

    However, since client 0's loader contains 90 % poisoned labels (attacker's
    target class), the CE term tends to reinforce the backdoor rather than
    preserve clean accuracy.  Clean accuracy is better restored in the
    subsequent post-training phase via FedAvg on clean clients (1–4).
    Therefore lambda_reg defaults to 0 (pure Ferrari loss).
"""

import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

from utils.model import get_model


def get_trigger_mask(dataset, device):
    """
    Automatically detect the backdoor trigger pixel positions by comparing
    a clean all-zero image with its backdoored version via ART's
    add_pattern_bd.  Returns a float tensor [C, H, W] where 1 = trigger pixel.
    """
    from art.attacks.poisoning.perturbations import add_pattern_bd

    if dataset == "mnist":
        h, w, c = 28, 28, 1
        # Pass a 2D array so ART uses its 2D branch (shape=(H,W)).
        # The 3D branch interprets the last dim as W, which causes an
        # IndexError when C=1 because distance+1 >= 1.
        clean_2d = np.zeros((h, w), dtype=np.float32)
        backdoored_2d = add_pattern_bd(clean_2d.copy())
        diff = np.abs(backdoored_2d - clean_2d)        # [H, W]
    else:  # cifar10 / cifar100
        h, w, c = 32, 32, 3
        clean = np.zeros((h, w, c), dtype=np.float32)
        backdoored = add_pattern_bd(clean.copy())
        diff = np.abs(backdoored - clean).sum(axis=-1)  # [H, W]

    trigger_map = (diff > 0).astype(np.float32)        # [H, W]

    # [H, W] → [C, H, W]
    mask = (
        torch.tensor(trigger_map, device=device)
        .unsqueeze(0)
        .expand(c, -1, -1)
        .contiguous()
        .float()
    )
    return mask


def unlearn(
    args, param, loader, sigma=0.1, n_samples=10, epochs=1, lr=1e-4, lambda_reg=0.0
):
    """
    Ferrari unlearning on the unlearning client's local data.

    Only the unlearning client (client 0) participates; the server simply
    accepts the returned model as the new global model.

    Args:
        args       : experiment config (args.dataset, args.device, etc.)
        param      : starting global model state dict (from case0)
        loader     : DataLoader for the unlearning client (client 0)
        sigma      : Gaussian noise std applied to trigger region
        n_samples  : Monte Carlo samples per mini-batch (paper uses 20;
                     reduce for faster CPU runs)
        epochs     : local optimisation epochs
        lr         : Adam learning rate (paper uses 1e-4)
        lambda_reg : weight for an optional CE regularisation term (default 0).
                     Keep at 0: client 0's labels are 90 % poisoned, so CE
                     reinforces the backdoor.  Clean accuracy is recovered by
                     the post-training phase (FedAvg on clean clients 1–4).

    Returns:
        (updated_state_dict, summary_dict)
    """
    model = get_model(args)
    model.load_state_dict(deepcopy(param))
    model.to(args.device)

    mask = get_trigger_mask(args.dataset, args.device)   # [C, H, W]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    total_samples = 0

    for _epoch in range(epochs):
        model.train()
        for data, labels in loader:
            data = data.to(args.device)      # [B, C, H, W]
            labels = labels.to(args.device)  # [B]
            B = data.size(0)

            optimizer.zero_grad()

            # Anchor: f(x) fixed — no gradient through original output.
            # This forces f_θ(x + δ_F) → f_θ(x), making the trigger
            # region have no effect on the prediction.
            with torch.no_grad():
                fx = model(data)          # [B, num_classes]

            # Monte Carlo estimate of feature sensitivity
            sens_list = []
            for _ in range(n_samples):
                # Perturbation confined to trigger pixels
                delta = torch.randn_like(data) * sigma * mask.unsqueeze(0)
                x_aug = (data + delta).clamp(0.0, 1.0)
                fx_aug = model(x_aug)     # [B, num_classes]  — gradient flows here

                out_diff = torch.norm(fx - fx_aug, dim=1)                  # [B]
                d_norm = (
                    torch.norm(delta.view(B, -1), dim=1).clamp(min=1e-8)   # [B]
                )
                sens_list.append((out_diff / d_norm).mean())

            ferrari_loss = torch.stack(sens_list).mean()

            # Optional CE regularisation (disabled by default, see docstring).
            if lambda_reg > 0.0:
                ce_loss = F.cross_entropy(model(data), labels)
                loss = ferrari_loss + lambda_reg * ce_loss
            else:
                loss = ferrari_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    avg_loss = total_loss / max(total_samples, 1)
    return (
        deepcopy(model.cpu().state_dict()),
        {"loss": avg_loss, "correct": 0, "total": total_samples},
    )
