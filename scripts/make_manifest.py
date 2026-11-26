"""Pair images and binary masks by relative stem and write a JSONL manifest."""
import argparse
import json
import os
from pathlib import Path


def by_stem(directory, suffixes):
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    index = {}
    for p in sorted(directory.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in suffixes:
            continue
        key = p.relative_to(directory).with_suffix("").as_posix()
        if key in index:
            raise ValueError(f"Duplicate relative stem {key!r} in {directory}")
        index[key] = p.resolve()
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    images = by_stem(args.images, {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"})
    masks = by_stem(args.masks, {".png", ".tif", ".tiff"})
    if not images or images.keys() != masks.keys():
        parser.error(f"Image and mask stems must match: {len(images.keys() - masks.keys())} "
                     f"without masks, {len(masks.keys() - images.keys())} without images")
    base = args.output.resolve().parent
    base.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for key in sorted(images):
            output.write(json.dumps({"image": os.path.relpath(images[key], base),
                                     "mask": os.path.relpath(masks[key], base)}) + "\n")
    print(f"Wrote {len(images)} pairs to {args.output}")


if __name__ == "__main__":
    main()
