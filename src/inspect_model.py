import json
import os
from transformers import AutoConfig

# ==========================================
# EXPERIMENT 01
# LLAMA 3.2 VISION ARCHITECTURE INSPECTION
# ==========================================

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

print("=" * 60)
print("LLAMA 3.2 VISION - MODEL INSPECTION")
print("=" * 60)

# 1. Load configuration only.
# This does NOT download the 11B model weights.

print("\nLoading model configuration...")

config = AutoConfig.from_pretrained(
    MODEL_ID,
    token=True
)

print("\nMODEL INFORMATION")
print("-" * 40)

print("Model type:", config.model_type)
print("Architecture:", config.architectures)
print("Model vocabulary size:", config.text_config.vocab_size)

# 2. Inspect vision encoder

print("\nVISION ENCODER CONFIGURATION")
print("-" * 40)

vision_config = config.vision_config.to_dict()

vision_parameters = [
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "image_size",
    "patch_size"
]

for key in vision_parameters:
    if key in vision_config:
        print(f"{key}: {vision_config[key]}")

# 3. Inspect language model

print("\nLANGUAGE MODEL CONFIGURATION")
print("-" * 40)

text_config = config.text_config.to_dict()

text_parameters = [
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "cross_attention_layers"
]

for key in text_parameters:
    if key in text_config:
        print(f"{key}: {text_config[key]}")

# 4. Save complete configuration

os.makedirs("results", exist_ok=True)

output_path = "results/model_config.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(config.to_dict(), file, indent=4)

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("Configuration saved to:", output_path)
print("=" * 60)
