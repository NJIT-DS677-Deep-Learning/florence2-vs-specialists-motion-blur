# Run DETR-ResNet-50 detection on COCO val2017, optionally with motion blur, and score it two ways.

# Mirrors src/run_florence_detection.py. Each run is scored with:
#   official: every prediction with DETR's own confidence (how DETR's published 42.0 AP was measured)
#   matched:  only confident predictions, all given score 1.0 (the same protocol as Florence-2, which has no scores)

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

from src.blur import apply_blur
from src.config import BLUR_SETTINGS, SEEDS
from src.data import load_val2017, open_image
from src.evaluate import score_detections, to_coco_box

MODEL_ID = "facebook/detr-resnet-50"
# Every blur level in config.py, so the choices never need updating here
BLUR_PX_CHOICES = sorted({px for levels in BLUR_SETTINGS.values() for px in levels})


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--limit", type=int)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--blur-type", choices=list(BLUR_SETTINGS), default="linear")
  parser.add_argument("--blur-px", type=int, choices=BLUR_PX_CHOICES, default=0)
  parser.add_argument("--seed", type=int, choices=SEEDS, default=0)
  # 0.925 gave DETR its highest matched mAP on the clean images (see 04_baseline_replication.ipynb, Step 7)
  parser.add_argument("--matched-threshold", type=float, default=0.925, help="Confidence a prediction needs to be kept in the matched protocol")
  args = parser.parse_args()

  items, coco = load_val2017()
  if args.limit is not None:
    if not 1 <= args.limit <= len(items):
      parser.error(f"--limit must be between 1 and {len(items)}")
    items = items[:args.limit]

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Loading {MODEL_ID} on {device}", flush=True)
  processor = DetrImageProcessor.from_pretrained(MODEL_ID)
  model = DetrForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()

  args.output_dir.mkdir(parents=True, exist_ok=True)
  official = []
  matched = []
  start = perf_counter()

  for index, item in enumerate(items, start=1):
    image_id = item["image_id"]
    # Open and blur the image (blur_px = 0 leaves it unchanged)
    with open_image(item["path"]) as image:
      image = apply_blur(image, args.blur_type, args.blur_px, args.seed, image_id)
      inputs = processor(images=image, return_tensors="pt").to(device)
      with torch.no_grad():
        output = model(**inputs)
      # Keep every prediction, with boxes in original image pixels
      result = processor.post_process_object_detection(output, threshold=0.0, target_sizes=[image.size[::-1]])[0]

    for score, label, box in zip(result["scores"].tolist(), result["labels"].tolist(), result["boxes"].tolist()):
      # DETR's class numbers are COCO category IDs (checked in the scoring notebook)
      detection = {"image_id": image_id, "category_id": int(label), "bbox": to_coco_box(box), "score": float(score)}
      official.append(detection)
      if score >= args.matched_threshold:
        matched.append({**detection, "score": 1.0})

    if index % 100 == 0 or index == len(items):
      print(f"Processed {index}/{len(items)} images", flush=True)

  inference_seconds = perf_counter() - start

  # Save predictions before scoring (compact JSON: the official list has 100 predictions per image)
  (args.output_dir / "official_predictions.json").write_text(json.dumps(official), encoding="utf-8")
  (args.output_dir / "matched_predictions.json").write_text(json.dumps(matched), encoding="utf-8")

  image_ids = [item["image_id"] for item in items]
  result = {
    "model": MODEL_ID,
    "split": "COCO val2017",
    "blur_type": args.blur_type,
    "blur_px": args.blur_px,
    "seed": args.seed,
    "n_images": len(items),
    "matched_threshold": args.matched_threshold,
    "inference_seconds": round(inference_seconds, 2),
    "official": {"n_detections": len(official), **score_detections(official, coco, image_ids)},
    "matched": {"n_detections": len(matched), **score_detections(matched, coco, image_ids)},
  }
  (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
