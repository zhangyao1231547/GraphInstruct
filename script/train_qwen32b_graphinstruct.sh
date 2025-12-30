#!/bin/bash
# Qwen3-32B GraphInstruct Text-Only Training Script
# Uses DeepSpeed ZeRO-3 for memory-efficient training on 4 GPUs

set -e

# Environment setup
PYTHON=/mnt/yrfs/GraphAgent_model/zy/llamafactory_env/bin/python
ACCELERATE=/mnt/yrfs/GraphAgent_model/zy/llamafactory_env/bin/accelerate

# GPU Configuration - use GPUs 4-7 (0-3 are occupied)
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false

# Model paths
BASE_MODEL=/mnt/yrfs/huggingface_models/Qwen3-32B
DATA_DIR=/mnt/yrfs/GraphAgent_model/zy/data/graphInstruct
OUTPUT_DIR=/mnt/yrfs/GraphAgent_model/train/qwen3-32b-graphinstruct

# Training hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=1
GRADIENT_ACCUMULATION=16
LEARNING_RATE=1e-5
MAX_LENGTH=4096
WARMUP_RATIO=0.03

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "============================================"
echo "Qwen3-32B GraphInstruct Training"
echo "============================================"
echo "Base Model: ${BASE_MODEL}"
echo "Data Dir: ${DATA_DIR}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Batch Size: ${BATCH_SIZE} x ${GRADIENT_ACCUMULATION} = $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Max Length: ${MAX_LENGTH}"
echo "============================================"

# Create accelerate config
ACCELERATE_CONFIG=/tmp/accelerate_config.yaml
cat > ${ACCELERATE_CONFIG} << 'EOF'
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_config_file: /tmp/ds_config_zero3.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# Create DeepSpeed config
cat > /tmp/ds_config_zero3.json << 'EOF'
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": 16,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOF

# Create training script
cat > /tmp/train_graphinstruct.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
Qwen3-32B GraphInstruct Training Script
Uses HuggingFace Trainer with DeepSpeed ZeRO-3
"""

import os
import sys
import json
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="/mnt/yrfs/huggingface_models/Qwen3-32B"
    )
    trust_remote_code: bool = field(default=True)


@dataclass
class DataArguments:
    data_dir: str = field(
        default="/mnt/yrfs/GraphAgent_model/zy/data/graphInstruct"
    )
    max_length: int = field(default=4096)


def load_graphinstruct_data(data_dir: str) -> List[Dict]:
    """Load GraphInstruct training data"""
    all_data = []

    # Load both English and Chinese data
    for filename in ["graphinstruct_english_text_190k.json", "graphinstruct_chinese_text_190k.json"]:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            logger.info(f"Loading {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.extend(data)
                logger.info(f"  Loaded {len(data)} samples")

    logger.info(f"Total samples: {len(all_data)}")
    return all_data


def preprocess_function(examples, tokenizer, max_length):
    """Preprocess examples for training"""
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for instruction, output in zip(examples["instruction"], examples["output"]):
        # Format as chat
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        # Create labels (mask user input)
        labels = tokenized["input_ids"].copy()

        # Find where assistant response starts and mask everything before
        user_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True
        )
        user_tokens = tokenizer(user_text, return_tensors=None)["input_ids"]

        # Mask user part with -100
        for i in range(min(len(user_tokens), len(labels))):
            labels[i] = -100

        model_inputs["input_ids"].append(tokenized["input_ids"])
        model_inputs["attention_mask"].append(tokenized["attention_mask"])
        model_inputs["labels"].append(labels)

    return model_inputs


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # Use SDPA instead of flash_attention_2
    )
    model.config.use_cache = False

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Load data
    logger.info(f"Loading data from {data_args.data_dir}")
    raw_data = load_graphinstruct_data(data_args.data_dir)

    # Convert to dataset
    dataset = Dataset.from_list(raw_data)
    logger.info(f"Dataset size: {len(dataset)}")

    # Preprocess
    logger.info("Preprocessing data...")
    processed_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, data_args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4,
        desc="Tokenizing",
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=data_args.max_length,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
PYTHON_SCRIPT

# Run training with DeepSpeed
echo "Starting training with DeepSpeed ZeRO-3..."

${ACCELERATE} launch \
    --config_file ${ACCELERATE_CONFIG} \
    /tmp/train_graphinstruct.py \
    --model_name_or_path ${BASE_MODEL} \
    --data_dir ${DATA_DIR} \
    --max_length ${MAX_LENGTH} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --bf16 True \
    --gradient_checkpointing True \
    --report_to none \
    --dataloader_num_workers 4 \
    --remove_unused_columns False

echo "============================================"
echo "Training Complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "============================================"
