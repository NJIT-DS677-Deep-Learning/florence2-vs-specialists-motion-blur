import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from src.data import load_karpathy_test, open_image
from src.evaluate import score_captions
from src.florence_inference import MODEL_ID, generate_caption, load_florence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    items = load_karpathy_test()
    if args.limit is not None:
        if not 1 <= args.limit <= len(items):
            parser.error(f"--limit must be between 1 and {len(items)}")
        items = items[:args.limit]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this on a GPU node.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {MODEL_ID} on {torch.cuda.get_device_name(0)}")
    model, processor, device = load_florence("cuda")

    predictions = {}
    references = {}
    start = perf_counter()

    for index, item in enumerate(items, start=1):
        image_id = item["image_id"]
        with open_image(item["path"]) as image:
            predictions[image_id] = generate_caption(
                model, processor, image, device
            )
        references[image_id] = item["references"]

        if index % 10 == 0 or index == len(items):
            print(f"Captioned {index}/{len(items)} images")

    inference_seconds = perf_counter() - start
    (args.output_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2), encoding="utf-8"
    )

    result = {
        "model": MODEL_ID,
        "split": "COCO Karpathy test",
        "blur_px": 0,
        "n_images": len(predictions),
        "inference_seconds": round(inference_seconds, 2),
        **score_captions(predictions, references),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
