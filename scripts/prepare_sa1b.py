"""Export one instance mask per SA-1B image and write train/val JSONL files."""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def decode_foreground(annotations, height, width):
    for annotation in sorted(annotations, key=lambda a: a["area"], reverse=True):
        rle = dict(annotation["segmentation"])
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, height, width)
        elif isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
        mask = mask_utils.decode(rle)
        if mask.shape != (height, width):
            raise ValueError(f"Mask dimensions {mask.shape} do not match {(height, width)}")
        if mask.any():
            return (mask > 0).astype(np.uint8) * 255
    return None


def is_validation(image_id, seed, fraction):
    key = hashlib.sha256(f"{seed}:{image_id}".encode("ascii")).digest()[:8]
    return int.from_bytes(key, "big") < int(fraction * 2**64)


def prepare(shard, output, max_images, val_fraction, seed):
    shard, output = shard.resolve(), output.resolve()
    if not shard.is_dir() or not 0 <= val_fraction < 1 or max_images <= 0:
        raise ValueError("Provide a shard directory, a positive image limit, and a validation fraction in [0, 1).")
    labels = sorted(shard.glob("sa_*.json"))
    if not labels:
        raise ValueError("No sa_*.json annotations found in the extracted shard.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "masks").mkdir()
    seen = set()
    counts = {"train": 0, "val": 0}
    with ExitStack() as stack:
        streams = {
            split: stack.enter_context((output / f"{split}.jsonl").open("x", encoding="utf-8"))
            for split in counts if split != "val" or val_fraction > 0
        }
        for label in labels:
            record = json.loads(label.read_text(encoding="utf-8"))
            info = record["image"]
            name = info["file_name"]
            image_id = int(info["image_id"])
            if Path(name).name != name or image_id < 0 or image_id in seen:
                raise ValueError(f"Invalid or duplicate image record: {label}")
            seen.add(image_id)
            image = shard / name
            width, height = int(info["width"]), int(info["height"])
            with Image.open(image) as rgb:
                if rgb.size != (width, height):
                    raise ValueError(f"Image dimensions do not match the annotation: {image}")
            mask = decode_foreground(record["annotations"], height, width)
            if mask is None:
                continue
            filename = f"sa_{image_id}.png"
            Image.fromarray(mask).save(output / "masks" / filename)
            split = "val" if is_validation(image_id, seed, val_fraction) else "train"
            streams[split].write(json.dumps({
                "image": os.path.relpath(image, output),
                "mask": f"masks/{filename}",
            }) + "\n")
            counts[split] += 1
            if sum(counts.values()) == max_images:
                break
    if counts["train"] == 0 or (val_fraction > 0 and counts["val"] == 0):
        raise ValueError(f"Empty split: {counts}. Retry in a new output directory with more images.")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=2000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(prepare(args.shard_dir, args.output_dir, args.max_images, args.val_fraction, args.seed))


if __name__ == "__main__":
    main()
