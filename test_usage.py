# -*- coding: utf-8 -*-
import os
import torch
from unsloth import FastLanguageModel
# Import standardize_sharegpt
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from datasets import load_from_disk # Load dataset from disk
from transformers import TextStreamer
import random
import traceback # For detailed error printing

# --- Configuration ---
base_model_load_dir = "./model"  # Directory where the original base model was saved
lora_model_load_dir = "./lora_model" # Directory where the LoRA adapters were saved
dataset_load_dir = "./dataset/finetome_100k_train" # Directory where the dataset was saved
max_seq_length = 2048
dtype = None # None for auto detection
load_in_4bit = True # Use 4bit quantization
num_test_samples = 5

# --- Load Fine-tuned Model and Tokenizer ---
print(f"Loading base model from {base_model_load_dir} and merging LoRA from {lora_model_load_dir}...")
if not os.path.exists(base_model_load_dir):
    raise FileNotFoundError(f"Base model directory not found: {base_model_load_dir}. Run download_resources.py first.")
if not os.path.exists(lora_model_load_dir):
    raise FileNotFoundError(f"LoRA model directory not found: {lora_model_load_dir}. Run fine_tune_model.py first.")

# Load the base model and tokenizer first
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_load_dir, # Load base model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Then merge the LoRA adapters
print("Applying LoRA adapters...")
try:
    model.load_adapter(lora_model_load_dir)
    print("LoRA adapters applied.")
except Exception as e:
    print(f"Error loading LoRA adapters from {lora_model_load_dir}: {e}")
    print("Ensure the directory contains adapter_config.json and adapter_model.safetensors (or .bin)")
    raise # Re-raise the exception

# --- Prepare for Inference ---
print("Preparing model for inference...")
FastLanguageModel.for_inference(model) # Enable native 2x faster inference (if applicable)

# Get the correct chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1", # Make sure this matches the fine-tuning template
)
# Ensure EOS token is set for padding if tokenizer doesn't have pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set tokenizer pad_token to eos_token.")

print("Model ready for inference.")

# --- Load Dataset for Test Samples ---
print(f"Loading dataset from {dataset_load_dir} to get test samples...")
if not os.path.exists(dataset_load_dir):
    raise FileNotFoundError(f"Dataset directory not found: {dataset_load_dir}. Run download_resources.py first.")

try:
    dataset = load_from_disk(dataset_load_dir)
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    if len(dataset) == 0:
        print("Dataset is empty. Cannot run inference.")
        exit()

    # ***** ADD STANDARDIZATION STEP HERE *****
    # Check if the first sample uses 'from'/'value' keys before standardizing
    if len(dataset) > 0 and isinstance(dataset[0].get('conversations'), list) and \
       len(dataset[0]['conversations']) > 0 and 'from' in dataset[0]['conversations'][0]:
        print("Dataset appears to be in ShareGPT ('from'/'value') format. Standardizing to 'role'/'content'...")
        dataset = standardize_sharegpt(dataset)
        print("Dataset standardized.")
    else:
        print("Dataset format appears compatible ('role'/'content') or is empty/invalid. Skipping standardization.")
    # *****************************************

except Exception as e:
    print(f"Error loading or standardizing dataset: {e}")
    raise

# --- Run Inference on Samples ---
print(f"\nRunning inference on {num_test_samples} random samples from the dataset...")

# Ensure we don't request more samples than available
num_samples_to_run = min(num_test_samples, len(dataset))
sample_indices = random.sample(range(len(dataset)), num_samples_to_run)

for i, idx in enumerate(sample_indices):
    print(f"\n--- Sample {i+1} (Index: {idx}) ---")
    try:
        sample = dataset[idx]
        # Now 'conversations' should always have 'role'/'content' if standardization worked
        original_conversations = sample.get('conversations')

        # Validate conversation data (should be standardized now)
        if not original_conversations or not isinstance(original_conversations, list) or len(original_conversations) == 0:
            print(f"Skipping sample {idx} - Invalid or empty 'conversations' list found after standardization attempt.")
            continue

        input_messages = []
        full_conversation_text_for_display = ""

        # Process the entire conversation history
        for msg_idx, msg in enumerate(original_conversations):
            role = msg.get('role') # Expect 'role' now
            content = msg.get('content') # Expect 'content' now
            # Basic validation for each message
            if role in ['user', 'assistant', 'system'] and isinstance(content, str) and content.strip(): # Allow 'system' role too
                # Add message if it's not an empty system prompt sometimes added by templates
                if not (role == 'system' and not content.strip()):
                    input_messages.append({"role": role, "content": content})
                    # Only display user/assistant turns for clarity
                    if role in ['user', 'assistant']:
                        full_conversation_text_for_display += f"**{role.capitalize()}**: {content}\n"
            else:
                # This warning should be less common now after standardization
                print(f"Warning: Skipping invalid message format at index {msg_idx} in sample {idx} even after standardization attempt: {msg}")

        # Check if we have any valid messages after processing
        if not input_messages:
             print(f"Skipping sample {idx} - No valid messages found after processing.")
             continue

        # Check the role of the *last valid message*
        last_original_message_role = input_messages[-1]['role']

        # If the conversation ended with an assistant, remove it to prompt for the next response.
        if last_original_message_role == 'assistant':
            input_messages.pop()
            # Adjust display text
            full_conversation_text_for_display = "\n".join(full_conversation_text_for_display.strip().split('\n')[:-1]).strip() + "\n"

        # Final check: Ensure input is not empty and ends with user or system prompt
        if not input_messages or input_messages[-1]['role'] not in ['user', 'system']:
             print(f"Skipping sample {idx} - Prepared input is empty or does not end with a user/system message after processing.")
             continue

        # --- Ready for Inference ---
        print(f"Input Conversation Context:\n{full_conversation_text_for_display}")
        print(f"**Assistant**:", end="") # Prompt for assistant's turn

        # Apply chat template and tokenize
        inputs = tokenizer.apply_chat_template(
            input_messages, # Use the processed list
            tokenize = True,
            add_generation_prompt = True, # Crucial for inference
            return_tensors = "pt",
        ).to(model.device)

        # Use TextStreamer for interactive output
        text_streamer = TextStreamer(tokenizer, skip_prompt = True)

        # Generate response
        with torch.no_grad():
            _ = model.generate(
                input_ids = inputs,
                streamer = text_streamer,
                max_new_tokens = 128,
                use_cache = True,
                temperature = 1.5,
                min_p = 0.1,
                pad_token_id=tokenizer.pad_token_id
            )
        print("\n---------------------------\n")

    except Exception as e:
        print(f"\n!!!!!!!!!!!!!! Error processing sample index {idx} !!!!!!!!!!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("Traceback:")
        traceback.print_exc()
        print("Sample Data (if available):", sample)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        continue

print("\nInference testing finished.")