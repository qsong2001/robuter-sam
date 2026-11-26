"""Run with: python -m unittest discover -s tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from segment_anything import sam_model_registry
from segment_anything.modeling import (
    ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer,
)
from train_lora import main


def tiny_sam(checkpoint=None):
    return Sam(
        image_encoder=ImageEncoderViT(img_size=64, patch_size=8,
                                     embed_dim=32, depth=1, num_heads=4, out_chans=32),
        prompt_encoder=PromptEncoder(embed_dim=32, image_embedding_size=(8, 8),
                                     input_image_size=(64, 64), mask_in_chans=16),
        mask_decoder=MaskDecoder(
            transformer_dim=32,
            transformer=TwoWayTransformer(depth=1, embedding_dim=32, num_heads=4, mlp_dim=64),
        ),
    )


class TestRelease(unittest.TestCase):
    def test_train_validate_and_save_on_synthetic_data(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rng = np.random.default_rng(0)
            for split in ("train", "val"):
                image = rng.integers(0, 256, size=(27, 43, 3), dtype=np.uint8)
                mask = np.zeros((27, 43), dtype=np.uint8)
                mask[3:20, 11:24] = 255
                Image.fromarray(image).save(root / f"{split}.png")
                Image.fromarray(mask).save(root / f"{split}_mask.png")
                manifest = {"image": f"{split}.png", "mask": f"{split}_mask.png"}
                (root / f"{split}.jsonl").write_text(json.dumps(manifest) + "\n")
            args = [
                "train_lora.py", "--sam-checkpoint", "unused.pth", "--device", "cpu",
                "--train-manifest", str(root / "train.jsonl"),
                "--val-manifest", str(root / "val.jsonl"),
                "--output", str(root / "adapter.pth"),
            ]
            with patch.dict(sam_model_registry, {"vit_b": tiny_sam}), patch("sys.argv", args):
                main()
            saved = torch.load(root / "adapter.pth", weights_only=True)
            self.assertEqual(len(saved["lora"]), 2 * len(saved["targets"]))
            self.assertTrue(all(n.endswith((".lora_A", ".lora_B")) for n in saved["lora"]))
            self.assertTrue(any(t.abs().max().item() > 0 for n, t in saved["lora"].items()
                                if n.endswith(".lora_B")))
