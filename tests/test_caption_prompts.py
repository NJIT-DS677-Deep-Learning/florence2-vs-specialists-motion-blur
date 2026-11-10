import torch
from PIL import Image
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.florence_inference import load_florence

image = Image.open(Path(__file__).parent / "test_image.jpg").convert("RGB")
model, processor, device = load_florence()

tasks = [
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
]

for task in tasks:
    inputs = processor(
        text=task,
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: (
            value.to(device=device, dtype=model.dtype)
            if key == "pixel_values"
            else value.to(device) if hasattr(value, "to") else value
        )
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            num_beams=3,
            do_sample=False,
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
    )[0]

    result = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=image.size,
    )

    print(f"\n{'=' * 60}")
    print("TASK:", task)
    print("OUTPUT:", result[task])