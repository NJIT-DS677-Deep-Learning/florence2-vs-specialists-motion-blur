import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from src.category_mapping import normalize_category
from src.data import load_val2017, open_image
from src.evaluate import score_detections, to_coco_box
from src.florence_inference import MODEL_ID, load_florence, detect_objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    items, coco = load_val2017()

    if args.limit is not None:
        if not 1 <= args.limit <= len(items):
            parser.error("Invalid image limit")
        items = items[:args.limit]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Map COCO class names to their official category IDs
    category_map = {
        cat["name"].lower(): cat["id"]
        for cat in coco.loadCats(coco.getCatIds())
    }

    print(f"Loading {MODEL_ID}...", flush=True)
    model, processor, device = load_florence("cuda")

    raw_predictions = []
    coco_predictions = []
    unmapped_labels = {}
    start = perf_counter()

    for index, item in enumerate(items, start=1):
        image_id = item["image_id"]

        with open_image(item["path"]) as image:
            output = detect_objects(
                model, processor, image, device
            )

        raw_predictions.append({
            "image_id": image_id,
            "detections": output
        })

        for box, label in zip(
            output["bboxes"], output["labels"]
        ):
            name = normalize_category(label)

            if name not in category_map:
                unmapped_labels[name] = (
                    unmapped_labels.get(name, 0) + 1
                )
                continue

            coco_box = to_coco_box(box)

            if coco_box[2] <= 0 or coco_box[3] <= 0:
                continue

            coco_predictions.append({
                "image_id": image_id,
                "category_id": category_map[name],
                "bbox": coco_box,
                "score": 1.0
            })

        if index % 10 == 0 or index == len(items):
            print(
                f"Processed {index}/{len(items)} images",
                flush=True
            )

    inference_seconds = perf_counter() - start

    # Save predictions before evaluation
    (args.output_dir / "raw_predictions.json").write_text(
        json.dumps(raw_predictions, indent=2),
        encoding="utf-8"
    )

    (args.output_dir / "coco_predictions.json").write_text(
        json.dumps(coco_predictions, indent=2),
        encoding="utf-8"
    )

    (args.output_dir / "unmapped_labels.json").write_text(
        json.dumps(unmapped_labels, indent=2),
        encoding="utf-8"
    )

    print("Unmapped labels:", unmapped_labels, flush=True)
    print("Valid detections:", len(coco_predictions), flush=True)

    metrics = score_detections(
        coco_predictions,
        coco,
        [item["image_id"] for item in items]
    )

    result = {
        "model": MODEL_ID,
        "split": "COCO val2017",
        "blur_px": 0,
        "n_images": len(items),
        "n_detections": len(coco_predictions),
        "scoring": "constant_score_1.0_with_9_aliases",
        "inference_seconds": round(inference_seconds, 2),
        **metrics
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
