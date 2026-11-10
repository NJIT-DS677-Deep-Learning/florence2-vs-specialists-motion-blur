
from pathlib import Path
from PIL import Image
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.florence_inference import (
    load_florence,
    generate_caption,
    detect_objects,
)

# Use an image stored locally in the project
image_path = Path(__file__).parent / "test_image.jpg"
image = Image.open(image_path).convert("RGB")

print("Loading Florence-2...")
model, processor, device = load_florence()
print(f"Running on: {device}")

print("\nGenerating caption...")
caption = generate_caption(model, processor, image, device)
print("Caption:", caption)

print("\nDetecting objects...")
detections = detect_objects(model, processor, image, device)
print("Detections:", detections)

if device == "cuda":
    print(
        "\nPeak GPU memory:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "GiB",
    )
