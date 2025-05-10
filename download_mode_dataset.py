# -*- coding: utf-8 -*-
import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from huggingface_hub import snapshot_download

# --- Configuration ---
model_name = "unsloth/Llama-3.2-3B-Instruct" # Or "unsloth/Llama-3.2-1B-Instruct"
dataset_name = "mlabonne/FineTome-100k"
model_save_dir = "./model"
dataset_save_dir = "./dataset"
# Set HF_HUB_ENABLE_HF_TRANSFER for potentially faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# --- Download Model and Tokenizer ---
print(f"Downloading model and tokenizer: {model_name}...")
try:
    # Use snapshot_download for potentially better handling of large models
    snapshot_download(repo_id=model_name, local_dir=model_save_dir, local_dir_use_symlinks=False)
    print(f"Model and tokenizer saved to {model_save_dir}")

    # Verify by trying to load (optional, but good practice)
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name = model_save_dir, # Load from local path
    #     max_seq_length = 2048,      # Specify necessary args
    #     dtype = None,               # Auto-detect
    #     load_in_4bit = True,        # Or False, depending on your setup
    # )
    # print("Model and tokenizer loaded successfully for verification.")
    # del model, tokenizer # Free up memory if verification was done

except Exception as e:
    print(f"Error downloading or verifying model/tokenizer: {e}")
    # Consider adding HF token if needed: token = "hf_..."

# --- Download Dataset ---
print(f"\nDownloading dataset: {dataset_name}...")
try:
    dataset = load_dataset(dataset_name, split="train")
    # Ensure the save directory exists
    os.makedirs(dataset_save_dir, exist_ok=True)
    # Save the dataset to disk
    dataset_path = os.path.join(dataset_save_dir, "finetome_100k_train")
    dataset.save_to_disk(dataset_path)
    print(f"Dataset saved to {dataset_path}")

    # Verify by trying to load (optional)
    # loaded_dataset = load_dataset(dataset_save_dir, split="train")
    # print(f"Dataset loaded successfully for verification. Size: {len(loaded_dataset)}")
    # del loaded_dataset # Free up memory

except Exception as e:
    print(f"Error downloading or saving dataset: {e}")

print("\nResource download process finished.")