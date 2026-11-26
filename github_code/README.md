# Robuster SAM — uniform LoRA fine-tuning

This repository provides a standalone, clean-image fine-tuning baseline for
box-prompted SAM (the original Segment Anything Model). It freezes pretrained
SAM and trains uniform-rank LoRA adapters in the image encoder and mask
decoder using BCE and Dice losses.

This release does not implement the paper's vulnerability profiling,
mixed-rank allocation, adversarial fine-tuning, or benchmark evaluation.
No data or model weights are included.

## Setup

Use a Python environment with PyTorch and TorchVision installed for your
hardware. Install the remaining dependencies and the upstream SAM package:

```bash
python -m pip install numpy pillow
python -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'
```

Download the matching SAM 1 checkpoint from the
[official SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints).

## Train

Create a JSON Lines manifest with one RGB image and one single-channel binary
foreground mask per example. Relative paths are resolved against the manifest
directory. The mask must match the image dimensions and contain a foreground
region (nonzero pixels). A tight box is derived from the mask.

```jsonl
{"image":"images/example.jpg","mask":"masks/example.png"}
```

```bash
python train_lora.py \
  --sam-checkpoint sam_vit_b_01ec64.pth \
  --train-manifest train.jsonl \
  --output outputs/uniform-r8.pth \
  --model-type vit_b --rank 8 --alpha 16 --epochs 1 --device cuda
```

The script trains on one image per optimizer update and saves only the adapter
tensors and their configuration. It does not overwrite an existing output.
Run `python train_lora.py --help` for more options.

To load an adapter, first build the same original SAM model from the same base
checkpoint, then apply the adapter layout and load the saved tensors:

```python
import torch
from segment_anything import sam_model_registry
from train_lora import add_uniform_lora

adapter = torch.load("outputs/uniform-r8.pth", map_location="cpu", weights_only=True)
sam = sam_model_registry[adapter["model_type"]](checkpoint="sam_vit_b_01ec64.pth")
assert add_uniform_lora(sam, adapter["rank"], adapter["alpha"]) == adapter["targets"]
assert set(adapter["lora"]) == {n for n, p in sam.named_parameters() if p.requires_grad}
sam.load_state_dict(adapter["lora"], strict=False)
sam.eval()
```
