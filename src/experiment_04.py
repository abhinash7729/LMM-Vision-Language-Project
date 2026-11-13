# ============================================================
# EXPERIMENT 04
# VISUAL QUESTION ANSWERING (VQA)
# MODEL: LLAMA 3.2 11B VISION INSTRUCT
# ============================================================

import json
import time
import torch
from datetime import datetime
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

RESULTS_PATH = (
    PROJECT_ROOT / "results" / "experiment_04_vqa_results.json"
)

# Edit these questions to match the uploaded image.
QUESTIONS = [
    "What are the main objects or people visible in this image?",
    "What activity or event is taking place in the image?",
    "Describe the background and surroundings of the image."
]

print("=" * 60)
print("EXPERIMENT 04 - VISUAL QUESTION ANSWERING")
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
    raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")

image = Image.open(IMAGE_PATH).convert("RGB")
image.thumbnail((1024, 1024))

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

# Configure generation tokens explicitly, following Experiment 03.
tokenizer = processor.tokenizer
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id

if pad_token_id is None:
    pad_token_id = eos_token_id

model.generation_config.eos_token_id = int(eos_token_id)
model.generation_config.pad_token_id = int(pad_token_id)

# ------------------------------------------------------------
# STEP 6: ASK QUESTIONS AND GENERATE ANSWERS
# ------------------------------------------------------------

results = []

for question_number, question in enumerate(QUESTIONS, start=1):
    print("\n" + "-" * 60)
    print(f"QUESTION {question_number}: {question}")
    print("-" * 60)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    print("Preprocessing image and question...")

    inputs = processor(
        image,
        prompt,
        add_special_tokens=False,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    print("Generating answer...")

    start_time = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            eos_token_id=int(eos_token_id),
            pad_token_id=int(pad_token_id)
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    execution_time = round(time.perf_counter() - start_time, 2)

    input_length = inputs["input_ids"].shape[-1]

    answer = processor.batch_decode(
        output[:, input_length:],
        skip_special_tokens=True
    )[0].strip()

    print("\nANSWER:")
    print(answer)
    print("Execution time:", execution_time, "seconds")

    results.append({
        "question_number": question_number,
        "question": question,
        "answer": answer,
        "execution_time_seconds": execution_time
    })

# ------------------------------------------------------------
# STEP 7: SAVE RESULTS AUTOMATICALLY
# ------------------------------------------------------------

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

experiment_results = {
    "experiment": "Experiment 04 - Visual Question Answering",
    "model": MODEL_ID,
    "execution_device": device,
    "gpu": torch.cuda.get_device_name(0)
           if torch.cuda.is_available() else "Not available",
    "input_image": str(IMAGE_PATH),
    "image_dimensions": image.size,
    "execution_date": datetime.now().isoformat(),
    "number_of_questions": len(QUESTIONS),
    "results": results
}

with open(RESULTS_PATH, "w", encoding="utf-8") as file:
    json.dump(experiment_results, file, indent=4, ensure_ascii=False)

print("\n" + "=" * 60)
print("EXPERIMENT 04 COMPLETED")
print("=" * 60)
print("All questions and answers saved successfully.")
print("Results location:", RESULTS_PATH)
