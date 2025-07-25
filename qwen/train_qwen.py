#!/usr/bin/env python3
import logging
import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

def main():
    parser = argparse.ArgumentParser(
        description="Finetune Qwen with DeepSpeed"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="Hugging Face model ID or local path"
    )
    parser.add_argument(
        "--train_file", type=str, required=True,
        help="A jsonl file with {'id', 'conversations': [{'from': 'user', 'value': ...}, {'from': 'assistant', 'value': ...}]} per line"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Where to write checkpoints"
    )
    parser.add_argument(
        "--deepspeed_config", type=str, required=True,
        help="Path to your deepspeed_config.json"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=1
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=10
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=1024
    )
    parser.add_argument(
        "--logging_steps", type=int, default=5
    )

    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training on {torch.cuda.device_count()} GPUs")

    # 1) TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eod_id

    # 2) MODEL
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()

    # 3) DATASET
    ds = load_dataset("json", data_files=args.train_file)
    train_ds = ds["train"]

    def tokenize_fn(examples):
        prompts = []
        for conversation in examples["conversations"]:
            prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            for turn in conversation:
                if turn["from"] == "user":
                    prompt += f"<|im_start|>user\n{turn['value']}<|im_end|>\n"
                elif turn["from"] == "assistant":
                    prompt += f"<|im_start|>assistant\n{turn['value']}<|im_end|>\n"
            prompts.append(prompt)

        # Tokenize
        out = tokenizer(
            prompts,
            truncation=True,
            max_length=args.max_seq_length,
            padding='max_length'
        )
        out["labels"] = out["input_ids"].copy()
        return out

    train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["id", "conversations"]
    )

    # 4) DATA COLLATOR
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5) TRAINING ARGS
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        include_num_input_tokens_seen=True,
        learning_rate=args.learning_rate,
        fp16=True,
        deepspeed=args.deepspeed_config,
        remove_unused_columns=True,
        report_to="wandb", 
    )

    # 6) TRAINER
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 7) LAUNCH
    trainer.train()
    # trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()