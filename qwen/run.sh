#!/bin/bash
deepspeed --hostfile hostfile \
  --num_nodes 1 \
  --num_gpus 1 \
  train_qwen.py \
  --model_name_or_path /data/models/qwen1.8b-chat/ \
  --train_file train_qwen.jsonl \
  --output_dir output \
  --deepspeed deepspeed_config.json \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 10 \
  --learning_rate 5e-5 \
  --max_seq_length 1024
