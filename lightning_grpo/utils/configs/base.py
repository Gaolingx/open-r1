"""Shared dataclass configuration for the Lightning GRPO pipeline."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Optional, get_args, get_origin, get_type_hints

import yaml


@dataclass
class PrecisionConfig:
    """Precision and numerics settings for Lightning training."""

    parameter_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    trainer_precision: Literal["bf16-mixed", "16-mixed", "16-true", "bf16-true", "32-true"] = "bf16-mixed"
    tf32: bool = True
    grad_scaler_enabled: bool = False


@dataclass
class LoRAConfig:
    """PEFT settings for adapter-based fine-tuning."""

    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    modules_to_save: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Model loading and architecture adaptation settings."""

    model_family: Literal["auto", "minimind"] = "auto"
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str = "main"
    tokenizer_name_or_path: Optional[str] = None
    model_config_path: Optional[str] = None
    model_init_kwargs: dict[str, Any] = field(default_factory=dict)
    pretrained_weight: str = "none"
    custom_weight_dir: str = "outputs"
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = "flash_attention_2"
    chat_template: Optional[str] = None
    eos_token: Optional[str] = None
    pad_token: Optional[str] = None
    freeze_embeddings: bool = False
    gradient_checkpointing: bool = True
    use_cache: bool = False
    compile_model: bool = False
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class DatasetSource:
    """One dataset source inside a mixture."""

    path: str
    config_name: Optional[str] = None
    split: str = "train"
    weight: float = 1.0
    columns: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """Dataset, formatting, and tokenization configuration."""

    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    train_split: str = "train"
    val_split: Optional[str] = None
    val_split_size: Optional[float] = None
    split_seed: int = 42
    prompt_column: str = "prompt"
    response_column: str = "response"
    messages_column: str = "messages"
    max_seq_length: int = 4096
    num_workers: int = 4
    preprocessing_batch_size: int = 256
    shuffle_buffer_size: int = 10000
    streaming: bool = False
    add_generation_prompt: bool = False
    mask_prompt_labels: bool = True
    assistant_only_loss: bool = False
    pack_sequences: bool = False
    preprocessing_use_cache: bool = True
    preprocessing_keep_in_memory: bool = False
    train_files: list[str] = field(default_factory=list)
    val_files: list[str] = field(default_factory=list)
    dataset_mixture: list[DatasetSource] = field(default_factory=list)
    text_column: str = "text"


@dataclass
class OptimizationConfig:
    """Optimizer, scheduler, and gradient update configuration."""

    optimizer: Literal["adamw", "adamw8bit"] = "adamw"
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8
    warmup_steps: int = 100
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    min_learning_rate: float = 0.0
    max_epochs: int = 1
    max_steps: int = -1
    gradient_accumulation_steps: int = 1
    gradient_clip_val: float = 1.0
    train_micro_batch_size: int = 1
    eval_micro_batch_size: int = 1


@dataclass
class LoggingConfig:
    """Experiment logging and sample inspection configuration."""

    project: str = "open-r1-lightning"
    run_name: str = "experiment"
    log_every_n_steps: int = 10
    flush_logs_every_n_steps: int = 50
    enable_wandb: bool = False
    enable_csv: bool = True
    sample_prompts: list[str] = field(default_factory=list)
    sample_every_n_steps: int = 0


@dataclass
class CheckpointConfig:
    """Model checkpointing configuration."""

    dirpath: str = "outputs"
    monitor: str = "train/loss"
    mode: Literal["min", "max"] = "min"
    save_top_k: int = 3
    save_last: bool = True
    every_n_train_steps: Optional[int] = 500


@dataclass
class EarlyStoppingConfig:
    """Early stopping behavior for monitored metrics."""

    enabled: bool = False
    monitor: Optional[str] = None
    mode: Optional[Literal["min", "max"]] = None
    patience: int = 3
    min_delta: float = 0.0
    check_finite: bool = True
    stopping_threshold: Optional[float] = None
    divergence_threshold: Optional[float] = None
    verbose: bool = False


@dataclass
class DistributedConfig:
    """Distributed strategy configuration for Lightning."""

    strategy: Literal["auto", "ddp", "fsdp"] = "auto"
    devices: int | str = "auto"
    num_nodes: int = 1
    accelerator: Literal["auto", "gpu", "cpu"] = "auto"
    sync_batchnorm: bool = False
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    fsdp_cpu_offload: bool = False
    fsdp_activation_checkpointing: bool = True
    fsdp_sharding_strategy: Literal["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"] = "FULL_SHARD"
    fsdp_auto_wrap_policy_classes: list[str] = field(default_factory=list)
    fsdp_activation_checkpointing_policy_classes: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """Top-level configuration consumed by Lightning entrypoints."""

    seed: int = 42
    task: Literal["sft", "grpo", "pretrain"] = "sft"
    output_dir: str = "outputs/default"
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert nested dataclasses to a serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from a YAML file."""

        with Path(path).open("r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle) or {}
        return cls._from_mapping(raw_config)

    @classmethod
    def _coerce_value(cls, annotation: Any, value: Any) -> Any:
        """Recursively convert mappings into typed dataclasses."""

        origin = get_origin(annotation)
        if origin in {list, tuple}:
            item_type = get_args(annotation)[0] if get_args(annotation) else Any
            if isinstance(value, list):
                return [cls._coerce_value(item_type, item) for item in value]
            return value

        if origin is not None:
            for candidate in get_args(annotation):
                if candidate is type(None):
                    continue
                coerced = cls._coerce_value(candidate, value)
                if coerced is not value or isinstance(value, candidate if isinstance(candidate, type) else tuple()):
                    return coerced
            return value

        if isinstance(annotation, type) and is_dataclass(annotation) and isinstance(value, dict):
            nested_type_hints = get_type_hints(annotation)
            coerced_values = {}
            for sub_field in fields(annotation):
                if sub_field.name not in value:
                    continue
                sub_annotation = nested_type_hints.get(sub_field.name, sub_field.type)
                coerced_values[sub_field.name] = cls._coerce_value(sub_annotation, value[sub_field.name])
            return annotation(**coerced_values)

        return value

    @classmethod
    def _from_mapping(cls, mapping: dict[str, Any]) -> "ExperimentConfig":
        kwargs: dict[str, Any] = {}
        type_hints = get_type_hints(cls)
        for field_info in fields(cls):
            if field_info.name not in mapping:
                continue
            value = mapping[field_info.name]
            if value is None:
                kwargs[field_info.name] = None
                continue
            annotation = type_hints.get(field_info.name, field_info.type)
            kwargs[field_info.name] = cls._coerce_value(annotation, value)
        return cls(**kwargs)
