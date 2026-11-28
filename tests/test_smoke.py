"""Run with: python -m unittest discover -s tests."""
import json
import contextlib
import io
import re
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
from train_lora import main, pgd_training_view


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
    def test_pgd_budget_and_input_only_gradient(self):
        model = tiny_sam().eval()
        from train_lora import add_uniform_lora
        add_uniform_lora(model, 2, 4)
        pixels = torch.full((3, 32, 64), 128.0)
        boxes = torch.tensor([[8.0, 4.0, 48.0, 24.0]])
        mask = np.zeros((32, 64), dtype=bool)
        mask[4:24, 8:48] = True
        view = pgd_training_view(model, pixels, boxes, mask, steps=2, epsilon=8.0)
        self.assertFalse(view.requires_grad)
        self.assertLessEqual((pixels - view).abs().max().item(), 8.0)
        self.assertTrue(torch.all((view >= 0) & (view <= 255)).item())
        self.assertFalse(torch.equal(view, pixels))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

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
                "--epochs", "2", "--pgd-steps", "2",
                "--eval-pgd-steps", "2",
            ]
            with patch.dict(sam_model_registry, {"vit_b": tiny_sam}), patch("sys.argv", args), \
                    patch("train_lora.pgd_training_view", wraps=pgd_training_view) as attack, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                main()
                self.assertEqual(attack.call_count, 1)
            scores = re.findall(r"val IoU ([0-9.]+)%, val PGD-2 IoU ([0-9.]+)%", output.getvalue())
            self.assertEqual(len(scores), 2)
            self.assertTrue(all(float(robust) <= float(clean) for clean, robust in scores))
            saved = torch.load(root / "adapter.pth", weights_only=True)
            self.assertEqual(saved["pgd_steps"], 2)
            self.assertEqual(len(saved["lora"]), 2 * len(saved["targets"]))
            self.assertTrue(all(n.endswith((".lora_A", ".lora_B")) for n in saved["lora"]))
            self.assertTrue(any(t.abs().max().item() > 0 for n, t in saved["lora"].items()
                                if n.endswith(".lora_B")))
