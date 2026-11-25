# Run BLIP-base captioning on the Karpathy test split, optionally with motion blur, and score it.

# Mirrors src/run_florence_captioning.py so both captioning models are run and scored the same way.

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.blur import apply_blur
from src.config import BLUR_SETTINGS, NUM_BEAMS, SEEDS
from src.data import load_karpathy_test, open_image
from src.evaluate import score_captions

MODEL_ID = "Salesforce/blip-image-captioning-base"
# Same upper limit as Florence-2 (src/florence_inference.py); captions normally end well before it
MAX_NEW_TOKENS = 128
# Every blur level in config.py, so the choices never need updating here
BLUR_PX_CHOICES = sorted({px for levels in BLUR_SETTINGS.values() for px in levels})


def caption_batch(model, processor, images, prompt, device, dtype, max_new_tokens):
  # Caption a list of images. A prompt (e.g. "a picture of") is removed from the returned captions.
  if prompt:
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt")
  else:
    inputs = processor(images=images, return_tensors="pt")
  # Move to the GPU; only the image tensor is converted to half precision
  inputs = inputs.to(device, dtype)

  with torch.no_grad():
    output_ids = model.generate(**inputs, num_beams=NUM_BEAMS, max_new_tokens=max_new_tokens)

  captions = processor.batch_decode(output_ids, skip_special_tokens=True)
  # BLIP repeats the prompt at the start of its caption: strip it so only the model's words are scored
  if prompt:
    captions = [c[len(prompt):].strip() if c.lower().startswith(prompt.lower()) else c.strip() for c in captions]
  return captions


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--limit", type=int)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--blur-type", choices=list(BLUR_SETTINGS), default="linear")
  parser.add_argument("--blur-px", type=int, choices=BLUR_PX_CHOICES, default=0)
  parser.add_argument("--seed", type=int, choices=SEEDS, default=0)
  # No prompt by default: in our tests "a picture of" lowered CIDEr by about 13 points and caused garbled captions
  parser.add_argument("--prompt", default="", help='Text BLIP continues from, e.g. "a picture of"; empty for none')
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
  # float16 by default (like Florence-2): it scored the same as float32 in our tests and runs about 3x faster
  parser.add_argument("--dtype", choices=["float16", "float32"], default="float16", help="Number precision on the GPU")
  args = parser.parse_args()

  items = load_karpathy_test()
  if args.limit is not None:
    if not 1 <= args.limit <= len(items):
      parser.error(f"--limit must be between 1 and {len(items)}")
    items = items[:args.limit]

  # Use the GPU if available; half precision only when asked for and on a GPU
  device = "cuda" if torch.cuda.is_available() else "cpu"
  dtype = torch.float16 if (args.dtype == "float16" and device == "cuda") else torch.float32
  print(f"Loading {MODEL_ID} on {device}", flush=True)
  processor = BlipProcessor.from_pretrained(MODEL_ID)
  model = BlipForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()

  args.output_dir.mkdir(parents=True, exist_ok=True)
  predictions = {}
  references = {}
  start = perf_counter()

  for first in range(0, len(items), args.batch_size):
    batch = items[first:first + args.batch_size]
    # Open and blur each image (blur_px = 0 leaves it unchanged)
    images = [apply_blur(open_image(item["path"]), args.blur_type, args.blur_px, args.seed, item["image_id"]) for item in batch]
    captions = caption_batch(model, processor, images, args.prompt, device, dtype, args.max_new_tokens)
    for item, caption in zip(batch, captions):
      predictions[item["image_id"]] = caption
      references[item["image_id"]] = item["references"]
    print(f"Captioned {len(predictions)}/{len(items)} images", flush=True)

  inference_seconds = perf_counter() - start
  (args.output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")

  result = {
    "model": MODEL_ID,
    "split": "COCO Karpathy test",
    "blur_type": args.blur_type,
    "blur_px": args.blur_px,
    "seed": args.seed,
    "prompt": args.prompt,
    "num_beams": NUM_BEAMS,
    "max_new_tokens": args.max_new_tokens,
    "dtype": str(dtype).replace("torch.", ""),
    "n_images": len(predictions),
    "inference_seconds": round(inference_seconds, 2),
    **score_captions(predictions, references),
  }
  (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
