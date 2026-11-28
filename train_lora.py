"""Base adversarial training with uniform-rank LoRA for box-prompted SAM 1."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Frozen linear projection plus a trainable rank-r residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.lora_A), self.lora_B)


def add_uniform_lora(sam, rank, alpha):
    if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Rank and alpha must be positive.")
    sam.requires_grad_(False)
    targets = []
    for name, layer in list(sam.named_modules()):
        if not isinstance(layer, nn.Linear):
            continue
        if not name.startswith(("image_encoder.", "mask_decoder.")):
            continue
        if name.startswith("mask_decoder.iou_prediction_head."):
            continue
        parent, _, child = name.rpartition(".")
        setattr(sam.get_submodule(parent), child, LoRALinear(layer, rank, alpha))
        targets.append(name)
    if not targets:
        raise ValueError("No SAM linear projections were matched.")
    return targets


def read_manifest(path):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or any(
            not isinstance(row.get(k), str) or not row[k] for k in ("image", "mask")
        ):
            raise ValueError(f"Invalid image/mask entry on manifest line {line_number}.")
        rows.append(tuple(path.parent / row[k] for k in ("image", "mask")))
    if not rows:
        raise ValueError("Training manifest is empty.")
    return rows


def read_pair(image_path, mask_path):
    with Image.open(image_path) as img:
        image = np.array(img.convert("RGB"), copy=True)
    with Image.open(mask_path) as img:
        mask = np.array(img, copy=True)
    if mask.ndim != 2 or image.shape[:2] != mask.shape:
        raise ValueError(f"Expected a single-channel mask matching {image_path}: {mask_path}")
    mask = mask > 0
    yy, xx = np.nonzero(mask)
    if xx.size == 0:
        raise ValueError(f"Mask {mask_path} is empty; a box cannot be inferred.")
    box = np.array([[xx.min(), yy.min(), xx.max() + 1, yy.max() + 1]], dtype=np.float32)
    return image, mask, box


def prepare_example(transform, image, box, device):
    resized = transform.apply_image(image)
    x = torch.as_tensor(np.array(resized, copy=True), device=device)
    boxes = torch.as_tensor(transform.apply_boxes(box, image.shape[:2]), device=device)
    return x.permute(2, 0, 1).contiguous().float(), boxes


def logits_from_pixels(sam, x, boxes, original_size):
    """Use the component APIs to keep gradients through pixels and adapters."""
    features = sam.image_encoder(sam.preprocess(x).unsqueeze(0))
    with torch.no_grad():
        sparse, dense = sam.prompt_encoder(points=None, boxes=boxes, masks=None)
    logits, _ = sam.mask_decoder(
        image_embeddings=features,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    return sam.postprocess_masks(logits, x.shape[-2:], original_size)


def mask_logits(sam, transform, image, box, device):
    x, boxes = prepare_example(transform, image, box, device)
    return logits_from_pixels(sam, x, boxes, image.shape[:2])


def segmentation_loss(logits, mask):
    target = torch.as_tensor(mask, dtype=logits.dtype, device=logits.device)[None, None]
    probs = logits.sigmoid()
    dice = 1 - (2 * (probs * target).sum() + 1) / (probs.sum() + target.sum() + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + dice


def pgd_training_view(sam, pixels, boxes, mask, *, steps, epsilon):
    """Maximize the segmentation loss in a raw-pixel L-infinity ball."""
    base = pixels.detach()
    low, high = (base - epsilon).clamp(0, 255), (base + epsilon).clamp(0, 255)
    view = (base + torch.empty_like(base).uniform_(-epsilon, epsilon)).clamp(0, 255)
    step_size = epsilon / steps
    for _ in range(steps):
        view = view.detach().requires_grad_(True)
        objective = segmentation_loss(logits_from_pixels(sam, view, boxes, mask.shape), mask)
        (gradient,) = torch.autograd.grad(objective, view)
        if not torch.isfinite(gradient).all().item():
            raise FloatingPointError("Nonfinite gradient in PGD.")
        with torch.no_grad():
            view = (view + step_size * gradient.sign()).maximum(low).minimum(high)
    return view.detach()


@torch.no_grad()
def evaluate_clean(sam, transform, pairs, device):
    """Mean per-image foreground IoU with ground-truth box prompts."""
    total_loss = total_iou = 0.0
    for image_path, mask_path in pairs:
        image, mask, box = read_pair(image_path, mask_path)
        logits = mask_logits(sam, transform, image, box, device)
        target = torch.as_tensor(mask, dtype=torch.bool, device=device)
        predicted = logits[0, 0] > 0
        total_loss += segmentation_loss(logits, mask).item()
        total_iou += ((predicted & target).sum() / (predicted | target).sum()).item()
    return total_loss / len(pairs), total_iou / len(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pgd-steps", type=int, default=3,
                        help="PGD iterations on alternating adversarial updates (0: clean only).")
    parser.add_argument("--epsilon", type=float, default=8.0,
                        help="L-infinity radius in raw pixel units [0, 255].")
    args = parser.parse_args()
    if args.epochs <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--epochs and --lr must be positive")
    if args.pgd_steps < 0 or not 0 < args.epsilon <= 255:
        parser.error("--pgd-steps must be nonnegative and --epsilon must be in (0, 255]")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide

    pairs = read_manifest(args.train_manifest)
    validation = read_manifest(args.val_manifest) if args.val_manifest else None
    if validation:
        train_images = {image.resolve() for image, _ in pairs}
        if any(image.resolve() in train_images for image, _ in validation):
            parser.error("Training and validation manifests share image paths")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    sam = sam_model_registry[args.model_type](checkpoint=str(args.sam_checkpoint))
    targets = add_uniform_lora(sam, args.rank, args.alpha)
    sam.to(device).eval()  # Keep the pretrained model's inference behavior; autograd remains on.
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    trainable = [p for p in sam.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    print(f"Adapting {len(targets)} projections ({sum(p.numel() for p in trainable):,} parameters).")

    update = 0
    for epoch in range(args.epochs):
        rng.shuffle(pairs)
        sums = {"clean": 0.0, "adv": 0.0}
        counts = {"clean": 0, "adv": 0}
        for image_path, mask_path in pairs:
            image, mask, box = read_pair(image_path, mask_path)
            optimizer.zero_grad(set_to_none=True)
            pixels, boxes = prepare_example(transform, image, box, device)
            kind = "clean"
            if args.pgd_steps and update % 2:
                pixels = pgd_training_view(sam, pixels, boxes, mask,
                                           steps=args.pgd_steps, epsilon=args.epsilon)
                kind = "adv"
            loss = segmentation_loss(logits_from_pixels(sam, pixels, boxes, mask.shape), mask)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite loss for {image_path}.")
            loss.backward()
            optimizer.step()
            sums[kind] += loss.item()
            counts[kind] += 1
            update += 1
        losses = ", ".join(f"{k} loss {sums[k] / counts[k]:.5f} ({counts[k]} updates)"
                           for k in ("clean", "adv") if counts[k])
        message = f"Epoch {epoch + 1}/{args.epochs}: {losses}"
        if validation:
            val_loss, val_iou = evaluate_clean(sam, transform, validation, device)
            message += f", val loss {val_loss:.5f}, val IoU {val_iou:.2%}"
        print(message, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adapters = {name: p.detach().cpu() for name, p in sam.named_parameters() if p.requires_grad}
    # Exclusive creation avoids accidentally overwriting a trained adapter.
    with args.output.open("xb") as stream:
        torch.save({"lora": adapters, "model_type": args.model_type, "rank": args.rank,
                    "alpha": args.alpha, "targets": targets,
                    "pgd_steps": args.pgd_steps, "epsilon": args.epsilon}, stream)
    print(f"Saved adapter weights to {args.output}")


if __name__ == "__main__":
    main()
