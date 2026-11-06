# lightning_grpo 用户手册

PyTorch Lightning 版的预训练 / SFT / DPO / GRPO 训练框架。所有任务都由 **一份 YAML 配置 + 一个入口脚本**驱动。

- [lightning\_grpo 用户手册](#lightning_grpo-用户手册)
  - [1. 安装环境](#1-安装环境)
  - [2. 目录结构](#2-目录结构)
  - [3. 运行四个 trainer](#3-运行四个-trainer)
    - [3.1 预训练](#31-预训练)
    - [3.2 SFT](#32-sft)
    - [3.3 DPO](#33-dpo)
    - [3.4 GRPO](#34-grpo)
  - [4. 配置速查](#4-配置速查)
  - [5. 数据集加载](#5-数据集加载)
    - [5.1 两种写法](#51-两种写法)
    - [5.2 各任务要求的列](#52-各任务要求的列)
    - [5.3 常用开关](#53-常用开关)
  - [6. 日志与 wandb](#6-日志与-wandb)
    - [wandb 登录](#wandb-登录)
    - [记录了什么](#记录了什么)
  - [7. 自定义模型加载](#7-自定义模型加载)
    - [两种加载方式](#两种加载方式)
    - [自定义 architecture（如 NekoMind）](#自定义-architecture如-nekomind)
    - [LoRA 注意事项](#lora-注意事项)
  - [8. 分布式与显存](#8-分布式与显存)
  - [9. 产物与断点续训](#9-产物与断点续训)
  - [10. 常见问题](#10-常见问题)

---

## 1. 安装环境

```bash
cd /path/to/open-r1
uv venv /path/to/venv --python 3.13
uv pip install --python /path/to/venv/bin/python -e .
```

关键依赖版本（已验证组合）：

| 包 | 版本 |
|---|---|
| torch | 2.11.0+cu128 |
| transformers | 5.17.0 |
| trl | 1.17.1 |
| lightning | 2.6.6 |
| peft | 0.21.0 |
| datasets | 5.0.1 |

其他可选：`liger-kernel`、`flash-attn`（`attn_implementation: flash_attention_2` 需要）、`wandb`。

> **不要**把 vLLM 装进训练环境。vLLM 会带来自己的一份 torch，可能降级训练用的 CUDA 构建。GRPO 的 vLLM rollout 请另建独立 venv（见 [grpo.md](grpo.md)）。

所有脚本都要求把 `src/` 放进 `PYTHONPATH`（`grpo` 的 reward 实现来自 `open_r1.rewards`）：

```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

或者用 `uv run --project .`。

---

## 2. 目录结构

```
lightning_grpo/
├── scripts/                 # 4 个训练入口
│   ├── train_pretrain.py
│   ├── train_sft.py
│   ├── train_dpo.py
│   └── train_grpo.py
├── config/training/         # 官方示例配置
│   └── custom/              # 你自己的配置都放这里
├── data/                    # 4 个 DataModule
├── models/                  # 4 个 LightningModule + 算法实现
├── strategies/              # DDP / FSDP2 / 张量并行
├── callbacks/               # checkpoint、日志、采样、NaN 检测
├── utils/configs/           # 配置 dataclass + YAML 加载器
└── docs/                    # 本文档目录
```

配置加载链路：`YAML` → `load_experiment_config()` → 按 `task` 字段选择 dataclass
（`pretrain` / `sft` / `dpo` / `grpo`）→ `dacite` 填充，**未知字段直接报错**，拼错 key 不会静默忽略。

---

## 3. 运行四个 trainer

四个脚本的公共参数：

| 参数 | 说明 |
|---|---|
| `--config` | **必填**，YAML 路径 |
| `--resume_from_checkpoint` | `.ckpt` 路径、目录，或 `last` |
| `--seed` | 覆盖 `config.seed` |
| `--precision` | `16`/`32`/`bf16`/`16-mixed`/`bf16-mixed` |
| `--gpus` | 覆盖 `distributed.devices`（整数） |
| `--lora_init_path` | 从已有 LoRA 权重热启动（自动开启 LoRA） |
| `--output_dir` | 覆盖 `output_dir` |

### 3.1 预训练

```bash
python lightning_grpo/scripts/train_pretrain.py \
  --config lightning_grpo/config/training/custom/pretrain_full_copy_1.yaml
```

- 数据：纯文本列（`data.text_column`，默认 `text`）
- 支持序列打包（`data.packing_enabled: true`，默认开）
- 支持 `liger_kernel.enabled: true`

### 3.2 SFT

```bash
python lightning_grpo/scripts/train_sft.py \
  --config lightning_grpo/config/training/custom/sft_lora_fsdp_copy.yaml
```

- 数据：OpenAI messages 列（`data.messages_column`，默认 `messages`）
- `data.completion_only_loss` / `assistant_only_loss` 控制只对 assistant 段算 loss
- 除公共参数外无额外 CLI 参数

### 3.3 DPO

```bash
python lightning_grpo/scripts/train_dpo.py \
  --config lightning_grpo/config/training/dpo_lora_fsdp.yaml \
  --beta 0.1 --loss_type sigmoid
```

- 数据：`chosen` / `rejected` 两列，均为 OpenAI messages；每侧**最后一条 assistant 消息**是 completion
- 额外参数：`--beta`（KL 系数）、`--loss_type`（`sigmoid` / `hinge` / `ipo`）

### 3.4 GRPO

```bash
python lightning_grpo/scripts/train_grpo.py \
  --config lightning_grpo/config/training/custom/grpo_qwen3_1_7b_math.yaml
```

GRPO 概念较多（prompt-only 数据、reward 函数、rollout 引擎、vLLM），单独见
**[grpo.md](grpo.md)**。

---

## 4. 配置速查

一个 YAML 的骨架：

```yaml
task: sft                 # pretrain | sft | dpo | grpo —— 决定用哪个 dataclass
seed: 42
output_dir: outputs/my_run
val_check_interval: 500   # 每 N 个优化步验证一次；0 = 不验证

precision:
  model_param_dtype: bf16     # 权重加载 dtype
  trainer_precision: bf16-true # Lightning precision
  tf32: true

model: { ... }            # 见第 7 节
data: { ... }             # 见第 5 节

optimization:
  optimizer: {type: adamw, learning_rate: 1.0e-5, betas: [0.9, 0.95], weight_decay: 0.01}
  scheduler: {type: cosine, warmup_steps: 100}
  max_epochs: 1
  max_steps: -1           # >0 时优先于 max_epochs
  gradient_accumulation_steps: 4
  gradient_clip_val: 1.0
  train_micro_batch_size: 2
  eval_micro_batch_size: 8

logging: { ... }          # 见第 6 节
checkpoint: { ... }       # 见第 9 节
distributed: { ... }      # 见第 8 节
```

可用的 `scheduler.type`：`linear`、`cosine`、`cosine_with_restarts`、`polynomial`、
`constant`、`constant_with_warmup`、`inverse_sqrt`、`reduce_lr_on_plateau`、
`cosine_with_min_lr`、`cosine_warmup_with_min_lr`、`warmup_stable_decay`。

---

## 5. 数据集加载

### 5.1 两种写法

**A. 单一数据集**（`data.dataset_name`）：

```yaml
data:
  dataset_name: parquet        # 或 json / arrow ...
  data_files:
    train: [/path/train-*.parquet]
    test:  [/path/val.parquet]
  train_split: train
  val_split: test              # 不填则自动找名为 test 的 split
```

`data_files` 支持三形态：单个字符串、字符串列表、`{split: path|path[]}` 字典。

**B. 数据集混合**（`data.dataset_list`），支持配比与分 split：

```yaml
data:
  dataset_list:
    - id: parquet
      split: train
      weight: 2.0             # 相对权重，全为 1.0 时走 concatenate，否则 interleave
      data_files: {train: [/a/*.parquet]}
    - id: json
      columns: [messages]     # 只保留这些列
      column_mapping: {conversation: messages}   # 源列名 -> 规范列名
      data_files: {train: [/b/*.jsonl]}
```

`split` 字段用于从该数据集中挑哪个 split 当 train（默认 `train`）；若同时存在
`data.val_split` 对应的 split，会一并作为验证集。

### 5.2 各任务要求的列

| task | 必需列 | 说明 |
|---|---|---|
| `pretrain` | `data.text_column`（默认 `text`） | 纯文本 |
| `sft` | `data.messages_column`（默认 `messages`） | `[{role, content}, ...]` |
| `dpo` | `chosen` / `rejected` | 同上格式，两条都是完整对话 |
| `grpo` | `messages` + reward 元数据 | 见 [grpo.md](grpo.md) |

`messages` **必须是真正的 list[dict]**，不接受 JSON 字符串或 ShareGPT 的
`conversations` 格式；列名不一致请用 `column_mapping` 在加载时改名。

### 5.3 常用开关

```yaml
data:
  max_seq_length: 4096          # 截断长度
  num_workers: 4                # preprocessing 与 dataloader worker 数
  preprocessing_batch_size: 128 # Dataset.map 的 batch
  preprocessing_use_cache: true # 复用 .map 缓存（改预处理逻辑后要清缓存）
  preprocessing_keep_in_memory: false
  streaming: false              # true 时用 IterableDataset（大数据集/不落盘）
  shuffle: true
  shuffle_buffer_size: 10000    # streaming 时的 buffer shuffle
  add_system_ratio: 0.0         # >0 时按比例随机插入内置 system prompt
  cache_dir: ./.cache/huggingface
```

> `streaming: true` 会关闭 sequence packing 以外的部分缓存能力，且 `num_workers` 强制为 0。

---

## 6. 日志与 wandb

```yaml
logging:
  project: open-r1-lightning
  run_name: my-run
  enable_csv: true          # 写 outputs/<run>/csv_logs/version_N/metrics.csv
  enable_wandb: true
  log_every_n_steps: 10
  # 训练中定期生成样本（可选）
  sample_prompts: ["介绍一下你自己"]
  sample_every_n_steps: 200
  sample_max_token: 128
```

### wandb 登录

```bash
wandb login            # 交互式粘贴 API key
# 或
export WANDB_API_KEY=xxxx
# 离线机器
wandb offline          # 之后 wandb sync
```

`wandb` 未安装时把 `enable_wandb` 设为 `false`，否则构建 logger 会失败。
国内网络慢可以：

```bash
export WANDB_BASE_URL=https://api.wandb.ai   # 或自建服务地址
export WANDB_MODE=offline
```

### 记录了什么

- 每步：`train/loss`、`train/reward`、`train/kl`、`train/entropy`、`train/grad_norm`、
  `perf/batch_time_seconds`、`perf/tokens_per_second`、`sample/global_samples`
- 每个 epoch：同名的 `*_epoch` 版本
- GRPO 额外：`train/rewards/<name>/mean`、`train/completions/mean_length`、
  `train/frac_reward_zero_std`、`train/cispo_clip_ratio`
- `sample_prompts` 非空且 `sample_every_n_steps > 0` 时，样本写入
  `csv_logs/version_N/sample_generations.csv`，同时作为 wandb Table 上传

CSV 与 wandb 里的 key 都带 `_step` / `_epoch` 后缀，这是 Lightning 的约定，监控
checkpoint 时用 `train/loss`（Lightning 会自动匹配）或写全 `train/loss_epoch`。

---

## 7. 自定义模型加载

```yaml
model:
  model_name_or_path: /abs/path/to/model
  tokenizer_name_or_path: /abs/path/to/tokenizer   # 可与模型不同
  model_generation_config_path: /abs/path/to/model # GRPO/SFT 采样用的 generation_config
  trust_remote_code: false
  attn_implementation: flash_attention_2           # flash_attention_2 | sdpa | eager
  gradient_checkpointing: true
  use_cache: false                                 # 有 gradient_checkpointing 时必须 false
  model_init_kwargs: {}                            # 直接 setattr 到 config，如 {rope_scaling: null}
  freeze_embeddings: false
  compile_model: false
  compile_kwargs: {}
  save_pth_format: true
  save_safetensors_format: false
  lora:
    enabled: true
    init_path: null
    r: 64
    alpha: 128
    dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
    modules_to_save: []
```

### 两种加载方式

| `custom_model` | 行为 | 适用 |
|---|---|---|
| `false`（推荐默认） | `AutoModelForCausalLM.from_pretrained(...)` | 绝大多数 HuggingFace 模型 |
| `true` | `AutoConfig.from_pretrained` → `from_config` → 手动分片 `load_state_dict` | 需要用 `model_init_kwargs` 改结构、或权重分片路径非标准 |

> ⚠️ `custom_model: true` 在**共享权重**模型上会失败（例如 Qwen3 的
> `tie_word_embeddings: true`，报 `Missing key(s): lm_head.weight`）。这类模型请用
> `custom_model: false`。

### 自定义 architecture（如 NekoMind）

仓库里的 `lightning_grpo/module/nekomind/` 提供了 NekoMindMoe2 的
`configuration_*.py` / `modeling_*.py`。有两种用法：

1. 模型目录下带 `auto_map`（`AutoConfig.register("nekomind_moe2", ...)` 的等价物），
   配上 `trust_remote_code: true`，用 `custom_model: false` 直接 `from_pretrained`。
2. 用 `custom_model: true` + `model_init_kwargs` 覆盖 config 字段，
   由 `build_configured_model_class()` 构造结构后手动灌权重。

### LoRA 注意事项

- 打开 `gradient_checkpointing` 时框架会自动调用 `enable_input_require_grads()`
- `--lora_init_path` 或 `model.lora.init_path` 可以从已有 adapter 继续训练

---

## 8. 分布式与显存

```yaml
distributed:
  strategy: fsdp2            # auto | ddp | fsdp2 | model_parallel
  accelerator: gpu
  devices: [0, 1]            # 列表 = 指定卡号；整数 = 前 N 张
  num_nodes: 1
  sync_batchnorm: false
  data_parallel_size: auto
  tensor_parallel_size: auto
  fsdp_reshard_after_forward: default   # default | never | always
  fsdp_cpu_offload: false
  fsdp_forward_prefetch: true
  fsdp_backward_prefetch: true
  fsdp_fully_shard_kwargs: {}
  tensor_parallel:
    enabled: false
    plan: auto
    sequence_parallel: false
    loss_parallel: false
```

- `strategy: fsdp2` / `model_parallel` 走 Lightning `ModelParallelStrategy`，
  分片在 `configure_model()` 里通过 `fully_shard` 完成
- `ddp` 适合小模型 / LoRA（全参数同步，显存占用高）
- `fsdp_reshard_after_forward: never` 在显存够时更快（少一次 all-gather）
- `--gpus N` 会覆盖 `devices`

显存不够时按顺序尝试：`gradient_checkpointing: true` → 降
`train_micro_batch_size` → 提高 `gradient_accumulation_steps` → `lora.r` 调小 →
`fsdp_reshard_after_forward: always`。

---

## 9. 产物与断点续训

`output_dir` 下的结构：

```
outputs/my_run/
├── resolved_config.json          # 实际生效的完整配置（排查配置问题先看这个）
├── csv_logs/version_N/           # metrics.csv, hparams.yaml, sample_generations.csv
├── checkpoints/
│   ├── last.ckpt/                # Lightning 分片 checkpoint
│   ├── model-epoch=..-step=...ckpt/
│   └── pt_checkpoint/*.pth       # 合并后的权重（save_pth_format）
└── hf_final/
    ├── pt_checkpoint/pretrain_model.pth
    └── hf_checkpoint/            # save_safetensors_format: true 时才有
```

注意：LoRA 训练导出的 state dict 是 **adapter 权重**（key 含 `lora_A`/`lora_B`），
需要 `PeftModel.from_pretrained(...).merge_and_unload()` 才能得到完整模型。

断点续训：

```bash
python lightning_grpo/scripts/train_sft.py --config cfg.yaml --resume_from_checkpoint last
python lightning_grpo/scripts/train_sft.py --config cfg.yaml --resume_from_checkpoint outputs/run/checkpoints
```

`optimization.resume_override.override_lr_on_resume` /
`reset_scheduler_on_resume` 控制在恢复时是否
复用旧的 lr 与 scheduler 状态。

---

## 10. 常见问题

**Q: `TypeError: Expected a GRPO config` / 未知字段报错**
配置里的 `task` 与所用脚本不匹配，或 YAML 有拼写错误的 key（加载器是严格模式）。
检查 `outputs/<run>/resolved_config.json`。

**Q: `ImportError: cannot import name 'get_kbit_device_map' from 'trl'`**
trl ≥ 1.0 删掉了该函数。`src/open_r1/utils/model_utils.py` 已做兼容回退；若你自行
升级了 trl 请确认这层回退还在。

**Q: 训练看起来卡住，日志不再增长**
先排除「在慢慢做验证」。`eval_micro_batch_size: 1` + HF `generate` 时，一条序列只有
~33 tok/s，50 条验证就要 10 分钟以上，而 CSV 只在验证循环结束后刷盘。把
`eval_micro_batch_size` 提到 8 左右。

需要抓栈时：

```bash
GRPO_DEBUG_STACKS=1 PYTHONPATH=src python lightning_grpo/scripts/train_grpo.py --config ...
kill -USR1 <pid>    # 每个进程写 /tmp/grpo_stack_<pid>.txt
```

**Q: `Missing key(s) in state_dict: "lm_head.weight"`**
`custom_model: true` 遇到了共享权重模型。改成 `custom_model: false`。

**Q: `std(): degrees of freedom is <= 0`**
batch 里只有一条样本（或一组奖励全相同）时算 std 的告警，来自指标聚合，
通常无害；把 `train_micro_batch_size` 调大即可消失。

**Q: 预处理结果没更新**
`preprocessing_use_cache: true` 会复用缓存。删掉 `cache_dir` 下对应的
`*.arrow`，或临时设为 `false`。

更多 GRPO 专属问题见 [troubleshooting.md](troubleshooting.md)。
