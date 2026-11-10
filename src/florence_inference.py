
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "microsoft/Florence-2-base-ft"

def load_florence(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="eager"
    ).to(device)

    model.eval()
    return model, processor, device


@torch.inference_mode()
def generate_caption(model, processor, image, device):
    """Generate a caption for a PIL image."""
    task = "<CAPTION>"
    inputs = processor(text=task, images=image, return_tensors="pt")
    inputs = {
        key: (
            value.to(device=device, dtype=model.dtype)
            if key == "pixel_values"
            else value.to(device) if hasattr(value, "to") else value
        )
        for key, value in inputs.items()
    }

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=3,
    )

    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]

    result = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=image.size,
    )

    return result[task]



@torch.inference_mode()
def detect_objects(model, processor, image, device):
    """Detect objects and return their labels and bounding boxes."""
    task = "<OD>"

    inputs = processor(
        text=task,
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: (
            value.to(device=device, dtype=model.dtype)
            if key == "pixel_values"
            else value.to(device) if hasattr(value, "to") else value
        )
        for key, value in inputs.items()
    }

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        num_beams=3,
    )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]

    result = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=image.size
    )

    return result[task]