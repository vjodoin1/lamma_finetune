#!/usr/bin/env python
"""LoRA fine‑tuning script for Llama‑2‑7B on the Alpaca dataset."""
import argparse, os
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
                          Trainer, DataCollatorForLanguageModeling)
from peft import get_peft_model, LoraConfig

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    return p.parse_args()

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds_path = os.path.join(args.data_dir, "alpaca_data.json")
    dataset = load_dataset("json", data_files=ds_path)["train"]

    def format_prompt(row):
        return {
            "text": f"### Instruction:\n{row['instruction']}\n### Input:\n{row['input']}\n### Response:\n{row['output']}"
        }

    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

    tokenized = dataset.map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=2048),
        batched=True, num_proc=4, remove_columns=["text"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_args = TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.batch_size // args.micro_batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        report_to="none",
    )

    Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=collator,
    ).train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()