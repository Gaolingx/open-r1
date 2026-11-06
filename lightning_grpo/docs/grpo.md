# GRPO 专项指南

GRPO = 对同一个 prompt 采样多条 completion，用组内相对优势做策略梯度。相比 SFT/DPO，
它多了三块东西：**prompt-only 数据 + reward 函数 + rollout 引擎**。

配置样例：`lightning_grpo/config/training/custom/grpo_qwen3_1_7b_math.yaml`

---

## 1. 数据必须是 prompt-only

GRPO 的输入只有「问题」，答案由模型自己生成。如果你直接拿 SFT/DPO 数据集
（`messages` 里已经带了 assistant 的金标回答），模型会在 prompt 里看到答案，训练完全失效。

`OpenR1-Math-220k` 就属于后者，需要先转换：

```bash
PYTHONPATH=src python -m lightning_grpo.scripts.prepare_grpo_math_dataset \
  --source /path/to/OpenR1-Math-220k/all \
  --glob 'train-*.parquet' \
  --output /path/to/OpenR1-Math-220k/grpo \
  --val-size 50 --seed 42
```

产物：

```
grpo/
├── train-00000.parquet   # 225079 行，已剔除 val 的 50 条
├── val.parquet           # 50 行
└── manifest.json
```

> ⚠️ `OpenR1-Math-220k/all/` 内部有**重复视图**：`train-*`（225129 行）是全集，
> `default-*` 和 `extended-*` 是它的子集。若用默认的 `--glob '*.parquet'` 会得到
> 45 万行、每条重复两遍。务必只选一组。

每行的列：

| 列 | 用途 |
|---|---|
| `messages` | `[{role: user, content: 题目}]`，渲染成 prompt |
| `solution` / `answer` | reward 的参考答案 |
| `uuid` / `problem_type` / `source` | 透传到 metadata，便于错误分析 |

**除 `messages` 以外的所有列**都会被打包进 `metadata`，再作为 kwargs 传给 reward 函数：

```yaml
data:
  dataset_name: parquet
  data_files:
    train: [/path/grpo/train-*.parquet]
    test:  [/path/grpo/val.parquet]
  train_split: train
  val_split: test
  add_generation_prompt: true
```

自定义数据集只要是「只有 user 轮 + 一个参考答案列」就能用。

---

## 2. reward 函数

`reward.reward_funcs` 从 `open_r1.rewards` 里按名字取：

| 名称 | 输入 | 说明 |
|---|---|---|
| `accuracy` | `completions`, `solution` | 用 `math_verify` 比对答案，解析失败返回 `None`（该样本被 nan 跳过） |
| `format` | `completions` | 严格匹配 `<think>…</think>\n<answer>…</answer>`，0/1 |
| `tag_count` | `completions` | 四个标签各 0.25，鼓励先学会输出结构 |
| `reasoning_steps` | `completions` | 检测分步推理，0~1 |
| `cosine` / `length` / `repetition_penalty` | — | 长度/重复类 shaping |

```yaml
reward:
  reward_funcs: [accuracy, format, tag_count]
  reward_weights: [1.0, 1.0, 1.0]   # 不填则全 1
  cosine_max_len: 1000
```

**关键前提**：`accuracy` 和 `format` 都要求模型输出 `<think>…</think><answer>…</answer>`。
Qwen3 base / Instruct 默认不这么输出，必须靠 `system_prompt` 约束（配置里已给）。
如果没有加 system prompt，`format`/`accuracy` 会长期恒为 0，只有 `tag_count` 有非零值。

---

## 3. rollout 引擎

### 3.1 torch（进程内 HF `generate`）

```yaml
rollout:
  engine: torch
  num_generations: 4
  max_completion_length: 2048
```

无需任何额外依赖，适合跑通 / 小规模验证。性能（L20，1.7B）：

| batch | 吞吐 |
|---|---|
| 1 | ~33 tok/s |
| 4 | ~135 tok/s |
| 8 | ~200 tok/s |

也就是 **bs 越大越划算**。`max_completion_length` 直接线性决定耗时：4 条 × 2048 token
单步约 110 s。

`TorchRolloutEngine` 会在生成期间临时打开 `use_cache`（训练本身仍是 `use_cache: false`），
否则解码是 O(n²)。

### 3.2 vLLM server（推荐用于正式训练）

需要一个独立进程，通过 HTTP 收发请求、通过 NCCL 同步权重。

```bash
# 1) 独立环境（不要装进训练环境）
uv venv /path/to/venv2 --python 3.13
uv pip install --python /path/to/venv2/bin/python -U vllm

# 2) 启动 server（推理卡与训练卡分开）
CUDA_VISIBLE_DEVICES=2,3 VLLM_SERVER_DEV_MODE=1 \
/path/to/venv2/bin/vllm serve /path/to/model \
  --weight-transfer-config '{"backend": "nccl"}' \
  --logprobs-mode processed_logprobs \
  --max-logprobs -1 \
  --enable-sleep-mode \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --tensor-parallel-size 2 \
  --port 8000
```

四个**不能省**的参数：

| 参数 | 原因 |
|---|---|
| `VLLM_SERVER_DEV_MODE=1` | `init_weight_transfer_engine` / `update_weights` 是 dev 接口 |
| `--weight-transfer-config '{"backend":"nccl"}'` | 训练侧用 NCCL 广播权重 |
| `--logprobs-mode processed_logprobs` | GRPO 需要 temperature/top-p 处理后的 logprob |
| `--max-logprobs -1` | 返回完整 top-k，供重要性比率使用 |

配置里切过去：

```yaml
rollout:
  engine: vllm
  vllm:
    mode: server
    server_base_url: http://127.0.0.1:8000
    server_timeout: 240.0
    group_port: 51216          # NCCL rendezvous 端口，双向放通
    enable_sleep_mode: true
    trust_remote_code: null    # null = 继承 model.trust_remote_code
    logprobs: 0
```

也可用 `mode: colocate`，vLLM 与训练同进程共卡（省机器，但抢显存）。

**数据面 vs 控制面**：

| 链路 | 协议 |
|---|---|
| `/health`、`/v1/models`、`/generate`、`/get_world_size` | HTTP |
| `/init_weight_transfer_engine`、`/update_weights` | HTTP（仅触发） |
| **实际权重张量** | **NCCL broadcast**（`group_port`） |

`optimizer_step()` 之后会自动 `update_policy()` → 推权重 → `reset_prefix_cache`。
只有 rank0 建 HTTP 连接，其它 rank 只参与 gather collective。

---

## 4. 算法与超参

```yaml
rollout:
  loss_type: cispo       # grpo | dapo | dr_grpo | cispo | gspo | sapo | dinpo | ...
  epsilon: 0.2
  epsilon_high: 5.0      # cispo 只裁上界，所以给很大的值
  kl_beta: 0.1
  use_reference_model: true
  temperature: 0.8
  top_p: 0.95
  max_prompt_length: 1024
  max_completion_length: 2048
  max_total_length: 3072
  num_generations: 4
  num_generations_eval: 1
```

`loss_type` 是一个**算法预设**，同时定下裁剪方式、优势估计器和 loss 聚合方式；
显式设置 `advantage_estimator` / `loss_agg_mode` 可以覆盖预设默认值。

⚠️ **`max_completion_length` 是 GRPO 最容易踩的坑**：如果答案被截断，
`<answer>` 闭合不了 → `accuracy=0`、`format=0` → 同一个 prompt 的 N 条 completion
奖励全相同 → 优势恒为 0 → 没有梯度。看指标：`train/frac_reward_zero_std` 长期为 1.0
就说明是这个情况。数学推理至少给 2048。

---

## 5. 指标解读

| 指标 | 期望 |
|---|---|
| `train/reward` | 逐步上升 |
| `train/frac_reward_zero_std` | 应 < 1.0；恒为 1 说明组内无差异 |
| `train/advantage_std` | 非 0 |
| `train/completions/clipped_ratio` | 不应长期为 1.0（说明全被截断） |
| `train/kl` | 缓慢增长；爆炸说明 `kl_beta` 太小或 lr 太大 |
| `train/entropy` | 不应快速掉到 0（策略坍缩） |
| `train/cispo_clip_ratio` | 过高说明 off-policy 太严重（`update_policy` 没生效） |

---

## 6. 调试

```yaml
rollout:
  debug_samples: true
  debug_every_n_steps: 20
```

直方图上会打印原始 prompt 与 completion —— 只用于确认 prompt 里没有泄漏答案、
模型确实在输出 `<think>/<answer>`。
