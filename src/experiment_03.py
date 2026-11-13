# ============================================================
# EXPERIMENT 03
# IMAGE UNDERSTANDING AND IMAGE CAPTION GENERATION
# MODEL: LLAMA 3.2 11B VISION INSTRUCT
# ============================================================

import os
import torch
from pathlib import Path
from PIL import Image

from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor
)

# ------------------------------------------------------------
# STEP 1: PROJECT CONFIGURATION
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

IMAGE_PATH = PROJECT_ROOT / "data" / "input_image.jpg"

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "experiment_03_caption.txt"
)

print("=" * 60)
print("EXPERIMENT 03 - IMAGE UNDERSTANDING")
print("=" * 60)

# ------------------------------------------------------------
# STEP 2: CHECK COMPUTING DEVICE
# ------------------------------------------------------------

if torch.cuda.is_available():
    device = "cuda"
    print("\nGPU:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("\nWARNING: GPU not available.")
    print("Using CPU for model execution.")

# ------------------------------------------------------------
# STEP 3: LOAD INPUT IMAGE
# ------------------------------------------------------------

print("\nLoading input image...")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Input image not found: {IMAGE_PATH}"
    )

image = Image.open(IMAGE_PATH).convert("RGB")

# Resize image while maintaining aspect ratio
image.thumbnail((1024, 1024))

print("Optimized image dimensions:", image.size)

print("Image loaded successfully.")
print("Image dimensions:", image.size)
print("Image mode:", image.mode)

# ------------------------------------------------------------
# STEP 4: LOAD HUGGING FACE PROCESSOR
# ------------------------------------------------------------

print("\nLoading Hugging Face processor...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Processor loaded successfully.")

# ------------------------------------------------------------
# STEP 5: LOAD LLAMA VISION MODEL
# ------------------------------------------------------------

print("\nLoading Llama 3.2 11B Vision model...")

model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True
)

model.eval()

print("Model loaded successfully.")

# ------------------------------------------------------------
# STEP 6: PREPARE IMAGE AND TEXT PROMPT
# ------------------------------------------------------------

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    "Describe this image in detail. "
                    "Identify the main objects, background, "
                    "colors, activities and overall scene."
                )
            }
        ]
    }
]

prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True
)

# ------------------------------------------------------------
# STEP 7: IMAGE PREPROCESSING
# ------------------------------------------------------------

print("\nPreprocessing image and text...")

inputs = processor(
    image,
    prompt,
    add_special_tokens=False,
    return_tensors="pt"
)

inputs = inputs.to(model.device)

print("Preprocessing completed successfully.")

print("\nINPUT TENSOR INFORMATION")
print("-" * 40)

for key, value in inputs.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: {value.shape}")

# Configure generation tokens explicitly
tokenizer = processor.tokenizer

eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id

if pad_token_id is None:
    pad_token_id = eos_token_id

print("EOS Token ID:", eos_token_id)
print("PAD Token ID:", pad_token_id)

# Ensure generation configuration uses normal integer IDs
model.generation_config.eos_token_id = int(eos_token_id)
model.generation_config.pad_token_id = int(pad_token_id)


# ------------------------------------------------------------
# STEP 8: GENERATE IMAGE DESCRIPTION
# ------------------------------------------------------------

print("\nGenerating image description...")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        eos_token_id=int(eos_token_id),
        pad_token_id=int(pad_token_id)
    )

print("\nImage description generated successfully!")

# Decode the generated response
input_length = inputs["input_ids"].shape[1]

generated_text = processor.batch_decode(
    output[:, input_length:],
    skip_special_tokens=True
)

print("\nIMAGE DESCRIPTION:")
print(generated_text[0])


# ------------------------------------------------------------
# STEP 9: DECODE GENERATED DESCRIPTION
# ------------------------------------------------------------

generated_tokens = output[
    :, inputs["input_ids"].shape[-1]:
]

caption = processor.batch_decode(
    generated_tokens,
    skip_special_tokens=True
)[0]

# ------------------------------------------------------------
# STEP 10: DISPLAY GENERATED CAPTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("GENERATED IMAGE DESCRIPTION")
print("=" * 60)

print(caption)

# ------------------------------------------------------------
# STEP 11: SAVE GENERATED RESULT
# ------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:
    file.write(caption)

print("\nCaption saved successfully.")

print("Output location:", OUTPUT_PATH)

# ------------------------------------------------------------
# EXPERIMENT COMPLETED
# ------------------------------------------------------------

print("\nEXPERIMENT 03 COMPLETED")
print("=" * 60)
