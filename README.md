# Robuster SAM

Code for **Robustifying Segment Anything through Attack-Conditioned
Low-Rank Capacity Allocation**.

This repository provides the **base version** of adversarial training for
box-prompted SAM 1 with uniform-rank LoRA. It freezes pretrained SAM and
optimizes uniform-rank adapters in the image encoder and mask decoder,
alternating one clean update with one adversarial update. The loss is BCE
plus Dice, and adversarial inputs are generated online with projected
gradient descent (PGD) and fixed box prompts.

This is a simplified training baseline, not the full paper implementation.
It does not include vulnerability profiling, mixed-rank allocation, the
paper's full attack suite, or benchmark evaluation. No data or model weights
are included.

## Repository layout

```text
robuter-sam/
├── train_lora.py                       # Training, validation, adapter export
├── requirements.txt
├── datasets/
│   ├── README.md                       # Image/mask and train/val layout
│   └── example.jsonl
├── scripts/
│   ├── make_manifest.py                # Build manifests from image/mask pairs
│   ├── prepare_sa1b.py                  # Decode SA-1B masks and write splits
│   └── prepare_coco_eval.py            # Prepare a fixed-box COCO evaluation set
├── tests/
│   └── test_smoke.py                   # Synthetic CPU tests
└── third_party/
    └── segment-anything/               # Original SAM 1 source and license
```

Place local model weights in **checkpoints/** and save trained adapters
under **outputs/**. Neither directory is shipped.

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

## Prepare your dataset

### Option A: Use SA-1B

1. Download a shard such as `sa000000` from the [official SA-1B dataset page](https://ai.meta.com/datasets/segment-anything/)
   under its dataset terms. Extract it locally; the shard should contain
   the images (e.g., `sa_1.jpg`) and their per-image COCO RLE annotations
   (e.g., `sa_1.json`) side by side. You do not need the entire dataset.
2. Install `pycocotools` and convert a subset:

```bash
python -m pip install pycocotools
python scripts/prepare_sa1b.py \
  --shard-dir datasets/sa1b_raw/sa000000 \
  --output-dir datasets/sa1b_demo \
  --max-images 2000 --val-fraction 0.1 --seed 0
```

The converter selects the largest nonempty annotated mask per image, saves it
as a binary PNG, and writes `train.jsonl` and `val.jsonl` with a
deterministic image-level split. Keep the extracted shard in place; the
manifests reference its images without copying them. Pass
`datasets/sa1b_demo/train.jsonl` and `datasets/sa1b_demo/val.jsonl`
to the training command below. This is an example base-version split,
not the paper's exact training and evaluation split.

### Option B: Use your own image/mask pairs

Put your dataset under `datasets/my_dataset/` like this:

```text
datasets/my_dataset/
├── train/
│   ├── images/scene001.jpg
│   └── masks/scene001.png
└── val/
    ├── images/scene101.jpg
    └── masks/scene101.png
```

Each RGB image needs one single-channel binary mask for the target object
(`0` is background; nonzero is foreground). Image and mask dimensions
must match. Use the same filename stem and relative subdirectory for each
pair. For example, `images/city/001.jpg` pairs with
`masks/city/001.png`. Split by image before exporting train/val masks:
the same image must not appear in both splits. For multi-object annotations,
export instance masks first; the trainer derives a tight box from each
nonempty mask.

The manifest is JSON Lines, one image/mask pair per line. All relative paths
are resolved from the manifest's directory. For example,
`datasets/my_dataset/train.jsonl` contains:

```jsonl
{"image":"train/images/scene001.jpg","mask":"train/masks/scene001.png"}
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

### Prepare evaluation data (COCO 2017)

Download **2017 Val images** and the **2017 Train/Val instance annotations**
from the [official COCO downloads](https://cocodataset.org/#download).
Extract `val2017/` and `annotations/instances_val2017.json` under
`datasets/coco2017/`. Install `pycocotools` and export evaluation pairs:

```bash
python -m pip install pycocotools
python scripts/prepare_coco_eval.py \
  --images-dir datasets/coco2017/val2017 \
  --instances-json datasets/coco2017/annotations/instances_val2017.json \
  --output-dir datasets/coco_eval \
  --max-images 128
```

The converter chooses one nonempty non-crowd instance per image, prioritizing
the largest annotated area. It exports binary PNG masks and an `eval.jsonl`
that points to the original images and records the selected image and
annotation IDs. By default it visits images in sorted ID order. To use your
own fixed list, replace `--max-images 128` with
`--image-ids-file path/to/image_ids.txt` (one ID per line).

Pass `--val-manifest datasets/coco_eval/eval.jsonl` to the training command.
The trainer derives tight ground-truth box prompts from the masks. Clean
validation and PGD validation start from these same inputs; adversarial
views are generated online against the model being evaluated, so no stored
adversarial images or separate attack cache is needed. This example is for
the base version, not the paper's full set of benchmark splits and protocols.

For another dataset, export RGB images and binary instance masks and use
`scripts/make_manifest.py` as above. Keep evaluation images out of the
training split.

## Train and evaluate

```bash
python train_lora.py \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --train-manifest datasets/my_dataset/train.jsonl \
  --val-manifest datasets/my_dataset/val.jsonl \
  --output outputs/uniform-r8.pth \
  --model-type vit_b --rank 8 --alpha 16 --epochs 1 --device cuda \
  --pgd-steps 3 --eval-pgd-steps 5 --epsilon 8
```

The script trains on one image per optimizer update and saves only the adapter
tensors and their configuration. PGD starts at a random perturbation and uses
step size epsilon divided by the number of steps. The default radius is 8
in the raw [0, 255] pixel scale (8/255 on [0, 1]); the perturbation is applied
after resizing and before SAM normalization and padding. Set
`--pgd-steps 0` to train on clean inputs only.

Training logs clean and adversarial loss separately. If a validation manifest
is supplied, it reports clean validation loss and IoU, plus PGD-5 IoU after
each epoch. The robust metric averages the worst IoU per image over the clean
view, one seeded random start, and all PGD iterates. Evaluation uses the same
perturbation budget and fixed ground-truth box prompts; it does not update
the model. Use `--eval-pgd-steps 0` to disable PGD validation without changing
the training attack. The script does not overwrite an existing output.
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
