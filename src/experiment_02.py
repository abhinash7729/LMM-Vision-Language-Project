import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
IMAGE_PATH = Path("data/sample_image.jpg")
OUTPUT_DIR = Path("results")

print("=" * 60)
print("EXPERIMENT 02 - IMAGE PREPROCESSING INSPECTION")
print("=" * 60)

# 1. Check the input image
if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}. "
        "Place sample_image.jpg inside the data folder."
    )

image = Image.open(IMAGE_PATH).convert("RGB")

print("\nINPUT IMAGE")
print("-" * 40)
print("Image format:", image.format)
print("Image dimensions:", image.size)
print("Image mode:", image.mode)

# 2. Load the model's processor, not the model weights
print("\nLoading image and text processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    token=True
)

print("Processor loaded successfully.")

# 3. Convert the image and prompt into model inputs
prompt = "<|image|>Describe the image and explain its visual features."

inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt"
)

print("\nPROCESSOR OUTPUT")
print("-" * 40)

tensor_shapes = {}

for key, value in inputs.items():
    if isinstance(value, torch.Tensor):
        tensor_shapes[key] = list(value.shape)
        print(f"{key}:")
        print("  Shape:", tuple(value.shape))
        print("  Data type:", value.dtype)

        if value.numel() > 0 and torch.is_floating_point(value):
            print("  Minimum:", value.min().item())
            print("  Maximum:", value.max().item())
    else:
        print(f"{key}: {type(value).__name__}")

# 4. Save experiment observations
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = {
    "experiment": "Experiment 02 - Image Preprocessing",
    "model": MODEL_ID,
    "input_image": str(IMAGE_PATH),
    "image_dimensions": list(image.size),
    "image_mode": image.mode,
    "prompt": prompt,
    "processor_tensor_shapes": tensor_shapes,
}

output_path = OUTPUT_DIR / "experiment_02_results.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\n" + "=" * 60)
print("EXPERIMENT 02 COMPLETED SUCCESSFULLY")
print("Results saved to:", output_path)
print("=" * 60)
