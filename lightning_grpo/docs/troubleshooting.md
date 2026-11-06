# 踩坑与排障

按「报错信息 → 原因 → 处理」组织。这些都是在本仓库真实遇到并修过的。

---

## 环境 / 依赖

### `ModuleNotFoundError: No module named 'trl.chat_template_utils'`

trl 0.18 没有这个模块，而 `models/grpo/tool_call.py` 需要它（tool calling 用）。

处理：升级到 `trl>=1.0`（仓库现在锁的是 1.17.1）。

### `ImportError: cannot import name 'get_kbit_device_map' from 'trl'`

trl ≥ 1.0 删掉了这个函数，但 `src/open_r1/utils/model_utils.py` 还在用。
它被 `open_r1.utils.__init__` 直接 re-export，所以任何 `import open_r1.rewards`
（GRPO 的 reward 来源）都会连带失败。

处理：已改为

```python
try:      # trl <= 0.18
    from trl import ModelConfig, get_kbit_device_map, get_quantization_config
except ImportError:  # trl >= 1.0
    from trl import ModelConfig, get_quantization_config
    def get_kbit_device_map(): ...
```

如果你升级/降级 trl 后出现同样报错，先确认这层回退还在。

### `ImportError: cannot import name 'load_causal_lm' from partially initialized module 'lightning_grpo.utils.modeling'`

在脚本里先 `import lightning_grpo.utils.modeling`、再导入依赖它的 module 会触发循环导入。

处理：先导入 `lightning_grpo.data.grpo_datamodule`（或直接 import
`lightning_grpo.models.grpo_module`），不要单独先 import `utils.modeling`。

### vLLM 与训练环境的 torch 冲突

vLLM 会安装自己的 torch。装进训练环境后 flash-attn / cu128 构建会被降级。

处理：训练环境（venv1）不装 vllm，另建 venv2 只跑 `vllm serve`。

---

## GRPO

### `TypeError: 'GRPORolloutConfig' object is not callable`

`LocalGenerateRolloutCoordinator.__init__` 曾用 `self.rollout = context.rollout`
遮蔽了同名的 `rollout()` 方法，第一个 `validation_step` 就炸。

处理：属性已重命名为 `self.rollout_config`。自定义子类时注意别再用 `self.rollout` 存配置。

### 训练进度停在 step 0 / 日志不再增长（看起来像死锁）

九成不是死锁，而是在**慢慢做验证**：

- `eval_micro_batch_size: 1` + HF `generate` → 单序列仅 ~33 tok/s
- 50 条验证集要串行跑 10 分钟以上
- CSV 只在验证循环结束后才刷盘，所以看起来「完全没动」

处理：`eval_micro_batch_size: 8`（bs=8 约 200 tok/s）。

确认没死锁 / 抓栈：

```bash
GRPO_DEBUG_STACKS=1 PYTHONPATH=src python lightning_grpo/scripts/train_grpo.py --config ...
# 另开终端
for p in $(pgrep -f train_grpo.py); do kill -USR1 $p; done
# 每个进程写 /tmp/grpo_stack_<pid>.txt
```

栈里如果停在 `generation/utils.py` 的 `_sample`，就是在正常采样，不是卡死。

> 多卡时不要用 `faulthandler` 直接写 stdout —— Lightning fork 出的 rank 共享同一个
> 文件句柄，输出会互相穿插成乱码。这也是上面的钩子按 pid 分文件的原因。

### reward 恒为 0.25、`advantage_std = 0`、`frac_reward_zero_std = 1`

说明组内 N 条 completion 奖励完全相同，优势为 0，没有梯度。常见原因：

1. `max_completion_length` 太小 → 全被截断（看 `clipped_ratio` 是否 = 1.0）
2. 没有 system prompt → 模型不输出 `<think>/<answer>` → `format=0`、`accuracy=0`
3. 数据集里残留了 assistant 金标 → prompt 泄漏答案
4. `temperature` 太低 → 采样几乎不随机

检查：

```yaml
rollout:
  debug_samples: true        # 打印 prompt/completion，确认不含答案、格式正确
```

### `Missing key(s) in state_dict: "lm_head.weight"`

`model.custom_model: true` 走了手动 `load_state_dict` 分片加载，而 Qwen3 的
`lm_head.weight` 与 `embed_tokens` 共享（`tie_word_embeddings: true`），分片里没有这个 key。

处理：`custom_model: false`。

### vLLM server 起不来 / 训练侧 `ConnectionError`

- `VLLM_SERVER_DEV_MODE=1` 忘了加 → `/init_weight_transfer_engine` 404
- `--weight-transfer-config '{"backend":"nccl"}'` 没配 → 权重同步无从建立
- `rollout.vllm.group_port` 被占用或防火墙拦了 → NCCL rendezvous 卡住
- 推理卡与训练卡跨机时，`group_port` 必须双向放通
- `server_timeout` 太小 → 大模型加载慢会提前抛 `ConnectionError`

### `std(): degrees of freedom is <= 0`

指标聚合里对只有 1 个元素的张量求 std 的无害告警（`train_micro_batch_size: 1`
或某个奖励函数对全部样本返回 `None` 时会出现）。调大 batch 即可。

---

## 数据

### `KeyError: Sample carries no non-empty 'messages' column`

该 split 的列名不是 `messages`。用 `column_mapping` 改名，或 `columns` 显式选择：

```yaml
dataset_list:
  - id: json
    column_mapping: {conversation: messages}
```

### 训练集行数是预期值的两倍

`OpenR1-Math-220k/all/` 里 `train-*` 是全集，`default-*` + `extended-*` 是它的子集。
用 `--glob 'train-*.parquet'` 只选一组。

### 预处理改动没生效

`preprocessing_use_cache: true` 复用了 `Dataset.map` 的 arrow 缓存。删掉
`cache_dir` 下对应文件，或临时设 `false`。

---

## 训练稳定性

### loss 变 NaN

内置 `NaNLossCallback` 会在检测到非有限 loss 时立即置 `should_stop`。
排查顺序：lr 太大 → `gradient_clip_val` 太小 → 数据里有空样本 → bf16 溢出（换
`precision.trainer_precision: bf16-true`）。

### 显存 OOM

按代价从低到高：`gradient_checkpointing: true` → 降 `train_micro_batch_size`、
升 `gradient_accumulation_steps` → `fsdp_reshard_after_forward: always` →
减 `rollout.max_completion_length` → 降 `lora.r` → `fsdp_cpu_offload: true`。

### `Found NN module(s) in eval mode at the start of training`

模型加载时调用了 `.eval()`（`load_causal_lm` 里的 reference model / 加载流程），
Lightning 在训练开始时提示存在处于 eval 模式的子模块。不影响训练，可忽略。
