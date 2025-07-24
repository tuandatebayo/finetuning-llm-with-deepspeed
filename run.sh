#!/bin/bash
deepspeed --hostfile hostfile \
  --num_nodes 1 \
  --num_gpus 1 \
  train.py \
  --model_name_or_path /data/models/llama3.2-1b/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08 \
  --train_file train.jsonl \
  --output_dir output \
  --deepspeed deepspeed_config.json \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 10 \
  --learning_rate 5e-5 \
  --max_seq_length 1024
