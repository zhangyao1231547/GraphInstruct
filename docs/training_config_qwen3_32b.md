# Qwen3-32B GraphInstruct 训练配置说明

## 服务器信息
- **服务器地址**: 10.0.50.13
- **GPU配置**: 8 × NVIDIA A800-SXM4-80GB
- **环境路径**: `/mnt/yrfs/GraphAgent_model/zy/llamafactory_env`

## 目录结构

```
/mnt/yrfs/GraphAgent_model/zy/
├── LLaMAFactory/                    # LLaMAFactory 代码库
│   ├── examples/
│   │   ├── deepspeed/
│   │   │   └── ds_z3_config.json   # DeepSpeed ZeRO-3 配置
│   │   └── train_reasoning/
│   │       ├── qwen3_32b_lora_sft_bilingual.yaml      # 原始4卡配置
│   │       ├── qwen3_32b_lora_sft_8gpu.yaml           # 8卡全量数据配置
│   │       └── qwen3_32b_lora_sft_8gpu_fast.yaml      # 8卡加速配置（数据减半）
│   └── data/
│       └── dataset_info.json        # 数据集注册信息
├── data/
│   └── graphInstruct/
│       ├── graphinstruct_english_text_190k.json      # 英文数据 (190k样本)
│       └── graphinstruct_chinese_text_190k.json      # 中文数据 (190k样本)
├── llamafactory_env/                # Conda 环境
└── model/                           # 模型存储

/mnt/yrfs/GraphAgent_model/train/
├── qwen3-32b-graphinstruct/         # 原4卡训练输出目录
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── checkpoint-1500/
│   ├── training.log
│   └── trainer_log.jsonl
└── qwen3-32b-graphinstruct-8gpu-fast/  # 8卡加速训练输出目录
    └── training.log

/mnt/yrfs/huggingface_models/
└── Qwen3-32B/                       # 基座模型
```

## 训练配置对比

### 原始4卡配置 (qwen3_32b_lora_sft_bilingual.yaml)
```yaml
model_name_or_path: /mnt/yrfs/huggingface_models/Qwen3-32B
finetuning_type: lora
lora_rank: 64
lora_target: all
deepspeed: examples/deepspeed/ds_z3_config.json
dataset: graphinstruct_english_190k,graphinstruct_chinese_190k
cutoff_len: 2048
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
# 总 batch size = 1 × 4卡 × 8 = 32
# 总步数 = 380000 / 32 × 3 epochs = 35,625
# 每步耗时 ~23秒
# 预估总时间 ~228小时 (9.5天)
```

### 8卡加速配置 (qwen3_32b_lora_sft_8gpu_fast.yaml)
```yaml
model_name_or_path: /mnt/yrfs/huggingface_models/Qwen3-32B
finetuning_type: lora
lora_rank: 64
lora_target: all
deepspeed: examples/deepspeed/ds_z3_config.json
dataset: graphinstruct_english_190k,graphinstruct_chinese_190k
max_samples: 95000                    # 每个数据集取95k，总共190k
cutoff_len: 2048
per_device_train_batch_size: 4        # 增大到4
gradient_accumulation_steps: 2        # 减少到2
# 总 batch size = 4 × 8卡 × 2 = 64
# 总步数 = 190000 / 64 × 3 epochs = 8,907
# 每步耗时 ~25秒
# 预估总时间 ~62小时 (2.5天)
```

## 启动命令

### 启动8卡加速训练
```bash
ssh 10.0.50.13

# 进入目录
cd /mnt/yrfs/GraphAgent_model/zy/LLaMAFactory

# 启动训练
nohup env PATH=/mnt/yrfs/GraphAgent_model/zy/llamafactory_env/bin:$PATH \
  /mnt/yrfs/GraphAgent_model/zy/llamafactory_env/bin/llamafactory-cli train \
  examples/train_reasoning/qwen3_32b_lora_sft_8gpu_fast.yaml \
  > /mnt/yrfs/GraphAgent_model/train/qwen3-32b-graphinstruct-8gpu-fast/training.log 2>&1 &
```

### 监控训练
```bash
# 查看实时日志
ssh 10.0.50.13 "tail -f /mnt/yrfs/GraphAgent_model/train/qwen3-32b-graphinstruct-8gpu-fast/training.log"

# 查看GPU状态
ssh 10.0.50.13 "nvidia-smi"

# 查看训练进度
ssh 10.0.50.13 "tail -20 /mnt/yrfs/GraphAgent_model/train/qwen3-32b-graphinstruct-8gpu-fast/training.log"
```

### 停止训练
```bash
ssh 10.0.50.13 "pkill -9 -f llamafactory"
```

## 数据集配置

数据集在 `/mnt/yrfs/GraphAgent_model/zy/LLaMAFactory/data/dataset_info.json` 中注册：

```json
{
  "graphinstruct_english_190k": {
    "file_name": "/mnt/yrfs/GraphAgent_model/zy/data/graphInstruct/graphinstruct_english_text_190k.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "graphinstruct_chinese_190k": {
    "file_name": "/mnt/yrfs/GraphAgent_model/zy/data/graphInstruct/graphinstruct_chinese_text_190k.json",
    "columns": {
      "prompt": "instruction",
      "response": "output"
    }
  }
}
```

## 训练参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| model_name_or_path | Qwen3-32B | 基座模型 |
| finetuning_type | lora | 使用LoRA微调 |
| lora_rank | 64 | LoRA秩 |
| lora_target | all | 对所有层应用LoRA |
| cutoff_len | 2048 | 最大序列长度 |
| learning_rate | 1e-4 | 学习率 |
| num_train_epochs | 3 | 训练轮数 |
| lr_scheduler_type | cosine | 余弦学习率调度 |
| warmup_ratio | 0.1 | 预热比例 |
| bf16 | true | 使用BF16精度 |
| gradient_checkpointing | true | 梯度检查点，节省显存 |
| deepspeed | ds_z3_config.json | DeepSpeed ZeRO-3 |

## 显存使用情况

| 配置 | 每卡显存使用 | 显存利用率 |
|------|-------------|-----------|
| 4卡 batch=1 | ~30GB | 37% |
| 8卡 batch=1 | ~20GB | 25% |
| 8卡 batch=4 | ~40GB | 50% |

## 注意事项

1. **checkpoint不能跨GPU数量恢复**: 4卡训练的checkpoint不能直接用于8卡训练恢复
2. **显存监控**: 如果显存不足，减小 `per_device_train_batch_size`
3. **训练中断恢复**: 在yaml中添加 `resume_from_checkpoint: /path/to/checkpoint`
4. **环境变量**: 必须设置PATH包含llamafactory_env/bin，否则torchrun找不到

## 更新记录

- 2025-12-25: 创建8卡加速训练配置，训练时间从9.5天缩短到2.5天
