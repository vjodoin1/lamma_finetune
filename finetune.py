# -*- coding: utf-8 -*-

import os
import torch
import pkg_resources # To check package versions
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from datasets import load_from_disk # Changed to load from disk
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer

# --- Configuration ---
model_load_dir = "./model"  # Directory where the model was saved
dataset_load_dir = "./dataset/finetome_100k_train" # Directory where the dataset was saved
lora_save_dir = "./lora_model" # Directory to save LoRA adapters
max_seq_length = 2048
dtype = None # None for auto detection
load_in_4bit = True # Use 4bit quantization

# LoRA configuration
lora_r = 16
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
lora_alpha = 16
lora_dropout = 0
bias = "none"
use_gradient_checkpointing = "unsloth" # True or "unsloth"
random_state = 3407

# Training arguments
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
warmup_steps = 5
max_steps = 60 # Set to None for a full epoch, adjust as needed
# num_train_epochs = 1 # Use this instead of max_steps for full run
learning_rate = 2e-4
weight_decay = 0.01
optim = "adamw_8bit"
lr_scheduler_type = "linear"
output_dir = "outputs" # Checkpoints directory

# --- Check TRL Version (Example Check) ---
try:
    trl_version = pkg_resources.get_distribution("trl").version
    print(f"Found TRL version: {trl_version}")
    # Add more checks or specific version requirements if needed
    # if pkg_resources.parse_version(trl_version) != pkg_resources.parse_version("0.15.2"):
    #     print("Warning: TRL version is not 0.15.2, which was used in the original notebook. Incompatibilities might arise.")
except pkg_resources.DistributionNotFound:
    print("Warning: TRL library not found or version could not be determined.")


# --- Load Model and Tokenizer ---
print(f"Loading model and tokenizer from {model_load_dir}...")
if not os.path.exists(model_load_dir):
    raise FileNotFoundError(f"Model directory not found: {model_load_dir}. Run download_resources.py first.")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_load_dir, # Load from local path
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
print("Model and tokenizer loaded.")

# --- Add LoRA Adapters ---
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_r,
    target_modules = lora_target_modules,
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout,
    bias = bias,
    use_gradient_checkpointing = use_gradient_checkpointing,
    random_state = random_state,
    use_rslora = False,
    loftq_config = None,
)
print("LoRA adapters added.")

# --- Load and Prepare Dataset ---
print(f"Loading dataset from {dataset_load_dir}...")
if not os.path.exists(dataset_load_dir):
    raise FileNotFoundError(f"Dataset directory not found: {dataset_load_dir}. Run download_resources.py first.")

dataset = load_from_disk(dataset_load_dir)
print("Dataset loaded.")

print("Processing dataset...")
# Set chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1", # Or the appropriate template for your model
)

# Standardize format if needed (assuming it's ShareGPT-like)
try:
    if "conversations" not in dataset.column_names or not isinstance(dataset[0]['conversations'], list):
         print("Dataset does not seem to be in ShareGPT format needing standardization. Skipping standardize_sharegpt.")
    elif isinstance(dataset[0]['conversations'][0], dict) and "from" in dataset[0]['conversations'][0]:
        print("Standardizing ShareGPT format...")
        dataset = standardize_sharegpt(dataset)
    else:
        print("Dataset format seems compatible, skipping standardization.")
except Exception as e:
    print(f"Could not standardize dataset, maybe it's already in the right format? Error: {e}")

# Formatting function
def formatting_prompts_func(examples):
    convos = examples["conversations"] # Assumes 'conversations' field exists
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

# Apply formatting
dataset = dataset.map(formatting_prompts_func, batched = True, num_proc=os.cpu_count() // 2 or 1)
print("Dataset processed.")


# --- Configure Trainer ---
print("Configuring SFTTrainer...")
training_args = TrainingArguments(
    per_device_train_batch_size = per_device_train_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    warmup_steps = warmup_steps,
    max_steps = max_steps,
    # num_train_epochs = num_train_epochs, # Use if max_steps is None
    learning_rate = learning_rate,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 1,
    optim = optim,
    weight_decay = weight_decay,
    lr_scheduler_type = lr_scheduler_type,
    seed = random_state,
    output_dir = output_dir,
    report_to = "none", # Change to "wandb", "tensorboard" etc. if needed
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer), # Important for padding
    dataset_num_proc = 2, # Adjust based on CPU cores
    packing = False, # Can be True for potentially faster training on short sequences
    args = training_args,
)

# Apply response-only training mask
print("Applying response-only training mask...")
trainer = train_on_responses_only(
    trainer,
    # Adjust these based on the actual chat template structure if needed
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
print("Trainer configured.")

# --- Train ---
print("Starting training...")
# Optional: Print memory stats before training
if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved before training.")
else:
    print("CUDA not available. Training on CPU.")
    start_gpu_memory = 0 # Assign a default value

# ***** THIS IS THE LINE WHERE THE ERROR OCCURRED *****
trainer_stats = trainer.train()
# ******************************************************

# Optional: Print memory and time stats after training
if torch.cuda.is_available():
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"\nTraining finished in {trainer_stats.metrics['train_runtime']:.2f} seconds.")
    print(f"Peak reserved memory = {used_memory} GB (+{used_memory_for_lora} GB for training).")
else:
    print(f"\nTraining finished in {trainer_stats.metrics['train_runtime']:.2f} seconds.")


# --- Save LoRA Adapters ---
print(f"\nSaving LoRA adapters to {lora_save_dir}...")
model.save_pretrained(lora_save_dir)
tokenizer.save_pretrained(lora_save_dir) # Save tokenizer alongside adapters
print("LoRA adapters saved.")

print("\nFine-tuning process finished.")