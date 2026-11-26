# Robuster SAM — uniform LoRA fine-tuning

Companion code for **Robustifying Segment Anything through Attack-Conditioned
Low-Rank Capacity Allocation**.

This repository provides a standalone, clean-image fine-tuning baseline for
box-prompted SAM (the original Segment Anything Model). It freezes pretrained
SAM and trains uniform-rank LoRA adapters in the image encoder and mask
decoder using BCE and Dice losses.

This release does not implement the paper's vulnerability profiling,
mixed-rank allocation, adversarial fine-tuning, or benchmark evaluation.
No data or model weights are included.

## Repository layout

- **train_lora.py** — clean-image training, optional validation, and adapter export.
- **scripts/make_manifest.py** — build JSONL splits from image and mask folders.
- **datasets/** — dataset layout and example manifest; place your own data here.
- **tests/** — synthetic CPU smoke test (no data or checkpoints required).
- **third_party/segment-anything/** — original SAM 1 Python source, pinned to
  an upstream revision, with the original license and attribution.

## Setup

Use a Python environment with PyTorch and TorchVision installed for your
hardware. Install the remaining dependencies and the bundled SAM package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e third_party/segment-anything
```

Download the matching SAM 1 checkpoint from the
[official SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints)
and save it under **checkpoints/**.

## Train

Create a JSON Lines manifest with one RGB image and one single-channel binary
foreground mask per example. Relative paths are resolved against the manifest
directory. See the [dataset layout](datasets/README.md) for the expected
train and validation folders. A tight box is derived from each mask.

```jsonl
{"image":"images/example.jpg","mask":"masks/example.png"}
```

To build manifests from the example directory structure:

```bash
for split in train val; do
  python scripts/make_manifest.py \
    --images "datasets/my_dataset/$split/images" \
    --masks "datasets/my_dataset/$split/masks" \
    --output "datasets/my_dataset/$split.jsonl"
done
```

```bash
python train_lora.py \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --train-manifest datasets/my_dataset/train.jsonl \
  --val-manifest datasets/my_dataset/val.jsonl \
  --output outputs/uniform-r8.pth \
  --model-type vit_b --rank 8 --alpha 16 --epochs 1 --device cuda
```

The script trains on one image per optimizer update and saves only the adapter
tensors and their configuration. If a validation manifest is supplied, it
logs clean mask loss and mean per-image foreground IoU after each epoch.
It does not overwrite an existing output.
Run `python train_lora.py --help` for more options.

Run the synthetic CPU test with `OMP_NUM_THREADS=1 python -m unittest discover -s tests`.

To load an adapter, first build the same original SAM model from the same base
checkpoint, then apply the adapter layout and load the saved tensors:

```python
import torch
from segment_anything import sam_model_registry
from train_lora import add_uniform_lora

adapter = torch.load("outputs/uniform-r8.pth", map_location="cpu", weights_only=True)
sam = sam_model_registry[adapter["model_type"]](checkpoint="checkpoints/sam_vit_b_01ec64.pth")
assert add_uniform_lora(sam, adapter["rank"], adapter["alpha"]) == adapter["targets"]
assert set(adapter["lora"]) == {n for n, p in sam.named_parameters() if p.requires_grad}
sam.load_state_dict(adapter["lora"], strict=False)
sam.eval()
```
