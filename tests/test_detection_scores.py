
import torch
from PIL import Image

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.florence_inference import load_florence

image = Image.open(Path(__file__).parent / "test_image.jpg").convert("RGB")
model, processor, device = load_florence()
task = "<OD>"

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

print("Generating detections with token scores...")

with torch.inference_mode():
    output = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

transition_scores = model.compute_transition_scores(
    sequences=output.sequences,
    scores=output.scores,
    beam_indices=output.beam_indices,
    normalize_logits=True,
)

text = processor.batch_decode(
    output.sequences,
    skip_special_tokens=False,
)[0]

detections = processor.post_process_generation(
    text,
    task=task,
    image_size=image.size,
)[task]

print("\nDetections found:", len(detections["labels"]))
print("Labels:", detections["labels"][:10])
print("Transition score shape:", tuple(transition_scores.shape))
print("First 10 token log-probabilities:")
print(transition_scores[0, :10].tolist())
print("Processor returns:", list(detections.keys()))

print("\nFirst 40 generated tokens and scores:")

token_ids = output.sequences[0].tolist()
tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)

print("Total sequence tokens:", len(tokens))
print("Scored tokens:", transition_scores.shape[1])

for i, token in enumerate(tokens[:40]):
    print(f"{i:3d}: {token}")


print("\nGenerated token / log-probability alignment:")

for i in range(1, min(41, len(tokens))):
    token = tokens[i]
    log_prob = transition_scores[0, i - 1].item()

    print(f"{i:3d}: {token:15s} log_prob={log_prob:.4f}")