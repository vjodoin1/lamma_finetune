#!/usr/bin/env python
"""Generate comparison prompts before and after fine‑tuning."""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def gen(model, tok, prompt):
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=128, do_sample=True, top_p=0.9)
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model_dir", required=True)
    a.add_argument("--ft_dir", required=True)
    a.add_argument("--out_file", required=True)
    args = a.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(args.model_dir, load_in_4bit=True, device_map="auto")
    ft   = AutoModelForCausalLM.from_pretrained(args.ft_dir,   load_in_4bit=True, device_map="auto")

    prompts = [
        "Translate to French: 'How are you today?'",
        "Write a haiku about the ocean.",
        "Summarize the plot of The Matrix in two sentences.",
        "List three benefits of exercise.",
        "Explain what a black hole is to a 12‑year‑old.",
    ]

    with open(args.out_file, "w") as f:
        for p in prompts:
            f.write("### Prompt\n" + p + "\n")
            f.write("## Base\n" + gen(base, tok, p) + "\n")
            f.write("## Fine‑tuned\n" + gen(ft, tok, p) + "\n\n")

if __name__ == "__main__":
    main()