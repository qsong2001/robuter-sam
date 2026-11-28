"""Export fixed-box evaluation masks from COCO instance annotations."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO


def prepare(images_dir, annotations, output, limit=None, requested_ids=None):
    images_dir, output = images_dir.resolve(), output.resolve()
    if not images_dir.is_dir() or (limit is not None and limit <= 0):
        raise ValueError("Provide an image directory and a positive --max-images.")
    coco = COCO(str(annotations))
    ids = sorted(coco.getImgIds()) if requested_ids is None else list(requested_ids)
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(coco.imgs):
        raise ValueError("Image IDs must be unique and present in the annotation file.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "masks").mkdir()
    count = 0
    with (output / "eval.jsonl").open("x", encoding="utf-8") as stream:
        for image_id in ids:
            info = coco.imgs[image_id]
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False))
            valid = sorted((a for a in anns if a.get("segmentation") and a.get("area", 0) > 0),
                           key=lambda a: (-a["area"], a["id"]))
            mask = chosen = None
            for ann in valid:
                candidate = coco.annToMask(ann)
                if candidate.any():
                    mask, chosen = candidate, ann
                    break
            if mask is None:
                if requested_ids is not None:
                    raise ValueError(f"No valid non-crowd mask for requested image {image_id}.")
                continue
            name = info["file_name"]
            if Path(name).name != name or Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(f"Unexpected COCO image filename: {name!r}")
            image_path = images_dir / name
            with Image.open(image_path) as image:
                if image.size != (info["width"], info["height"]):
                    raise ValueError(f"Image dimensions disagree with the annotations: {image_path}")
            if mask.shape != (info["height"], info["width"]):
                raise ValueError(f"Mask dimensions disagree with the annotations: {image_id}")
            target = output / "masks" / f"{image_id:012d}.png"
            Image.fromarray((mask > 0).astype(np.uint8) * 255).save(target)
            stream.write(json.dumps({"image": os.path.relpath(image_path, output),
                                     "mask": target.relative_to(output).as_posix(),
                                     "image_id": image_id, "annotation_id": chosen["id"]}) + "\n")
            count += 1
            if limit is not None and count == limit:
                break
    if not count:
        raise ValueError("No evaluation examples with valid masks.")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--instances-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--max-images", type=int)
    group.add_argument("--image-ids-file", type=Path, help="Plain text with one COCO image ID per line.")
    args = parser.parse_args()
    ids = ([int(line) for line in args.image_ids_file.read_text().splitlines() if line.strip()]
           if args.image_ids_file else None)
    count = prepare(args.images_dir, args.instances_json, args.output_dir, args.max_images, ids)
    print(f"Wrote {count} COCO examples to {args.output_dir / 'eval.jsonl'}")


if __name__ == "__main__":
    main()
