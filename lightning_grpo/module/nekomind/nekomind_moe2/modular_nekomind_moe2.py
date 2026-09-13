# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch NekoMind model."""
import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention
from ..glm5_next.modeling_glm5_next import (
    Glm5NextTextForgetGate,
    Glm5NextTextLinearAttention,
    Glm5NextTextRMSNormGated,
)
from ..llama.modeling_llama import LlamaRMSNorm, eager_attention_forward
from ..mixtral.modeling_mixtral import MixtralExperts, load_balancing_loss_func
from ..gemma.modeling_gemma import GemmaMLP


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nekocyrene/NekoMind1.5-Base")
@strict
class NekoMindMoe2Config(PreTrainedConfig):
    r"""
    mla_use_nope (`bool`, *optional*, defaults to `True`):
        Use NoPE attention. MLA asserts that this is enabled.
    mla_use_output_gate (`bool`, *optional*, defaults to `False`):
        Apply sigmoid output gate before the MLA output projection.
    mlp_layer_types (`list[str]`, *optional*):
        List of layer types for the MLP or MoE layers. Defaults to None.
    linear_head_dim (`int`, *optional*):
        Dimension of each head in linear attention layers. Defaults to 128.
    linear_num_heads (`int`, *optional*):
        Number of heads for the linear attention layers. Defaults to 32.
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size for the short convolution applied to queries, keys, and values in linear attention layers.

    ```python
    >>> from transformers import NekoMindMoeModel2, NekoMindMoeConfig2

    >>> # Initializing a NekoMindMoE style configuration
    >>> configuration = NekoMindMoeConfig2()

    >>> # Initializing a model from the NekoMind1.5-Base" style configuration
    >>> model = NekoMindMoeModel2(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nekomind_moe2"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_token",
    }

    # Default tensor parallel plan for base model `NekoMindMoe`
    base_model_tp_plan = {
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int | None = 128
    mla_use_nope: bool = True
    mla_use_output_gate: bool = False
    linear_lower_bound: float | None = -5.0
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    attention_dropout: float | int = 0.0
    attention_bias: bool = False
    moe_intermediate_size: int = 768
    shared_expert_intermediate_size: int = 768
    num_experts_per_tok: int = 8
    num_experts: int = 128
    norm_topk_prob: bool = False
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    mlp_layer_types: list[str] | None = None
    layer_types: list[str] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    linear_head_dim: int = 128
    linear_num_heads: int = 32
    linear_conv_kernel_dim: int = 4

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim

        super().__post_init__(**kwargs)
        # Checkpoint stores linear attention attributes in a config sub-dict: if it's there, extract them
        linear_attn_config = kwargs.get("linear_attn_config", {})
        self.linear_head_dim = linear_attn_config.get("head_dim", self.linear_head_dim)
        self.linear_num_heads = linear_attn_config.get("num_heads", self.linear_num_heads)
        self.linear_conv_kernel_dim = linear_attn_config.get("short_conv_kernel_size", self.linear_conv_kernel_dim)
        self.linear_lower_bound = linear_attn_config.get("gate_lower_bound", self.linear_lower_bound)

        # For layer types, the precedence is: checkpoint config > layer types > default
        if self.layer_types is None:
            if "full_attn_layers" in linear_attn_config and "kda_layers" in linear_attn_config:
                layer_types = [None] * self.num_hidden_layers
                for layer in linear_attn_config["full_attn_layers"]:
                    layer_types[layer - 1] = "full_attention"  # types are 1-indexed in the checkpoint
                for layer in linear_attn_config["kda_layers"]:
                    layer_types[layer - 1] = "linear_attention"
                self.layer_types = layer_types
            else:
                self.layer_types = [
                    "full_attention" if i and i % 4 == 0 else "linear_attention" for i in range(self.num_hidden_layers)
                ]

        # Same for MLP layer types, which indicate MLP or MoE
        if self.mlp_layer_types is None:
            first_k_dense_replace = kwargs.get("first_k_dense_replace", 1)
            self.mlp_layer_types = [
                "dense" if i < first_k_dense_replace else "sparse" for i in range(self.num_hidden_layers)
            ]


class NekoMindMoe2Attention(DeepseekV3Attention):
    """Multi-headed Latent Attention (MLA) from Deepseek V2 with NoPE, but the part of the keys where RoPE is applied is
    still shared."""

    def __init__(self, config: NekoMindMoe2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.scaling = self.qk_head_dim ** (-0.5)

        self.use_output_gate = config.mla_use_output_gate
        if self.use_output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.v_head_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        query_states = q_states.view(query_shape).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_nope = self.kv_a_layernorm(kv_nope)
        # Both latents are viewed as single-head, 4D tensors so all cache layers handle them correctly
        kv_nope = kv_nope.view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        # Cache read / write is performed while latent KV is still compressed
        if past_key_values is not None:
            kv_nope, k_rot = past_key_values.update(kv_nope, k_rot, self.layer_idx)

        key_states, value_states = self.expand_kv(kv_nope, k_rot)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        # Divergence from Deepseek V2 MLA: there is an additional gate to control the output of the attention
        if self.use_output_gate:
            attn_output = attn_output * self.g_proj(hidden_states).sigmoid()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class NekoMindMoe2ForgetGate(Glm5NextTextForgetGate):

    def __init__(self, config: NekoMindMoe2Config):
        super().__init__(config)


class NekoMindMoe2DeltaAttention(Glm5NextTextLinearAttention):
    """Kimi Linear Attention: this is essentialy the same a gated delta net (GDN) but decay is per-channel instead of
    per-token."""

    def __init__(self, config: NekoMindMoe2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.forget_gate = NekoMindMoe2ForgetGate(config)
        self.o_norm = NekoMindMoe2RMSNormGated(self.head_dim, eps=self.layer_norm_epsilon)


class NekoMindMoe2MLP(GemmaMLP):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]


class NekoMindMoe2Experts(MixtralExperts):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_experts
        self.intermediate_dim = config.moe_intermediate_size


class NekoMindMoe2TopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        routing_weights = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)  # (seq_len, top_k)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_scores = router_top_value.to(routing_weights.dtype)
        return router_logits, router_scores, router_indices


class NekoMindMoe2SparseMoeBlock(nn.Module):
    def __init__(self, config: NekoMindMoe2Config):
        super().__init__()
        self.experts = NekoMindMoe2Experts(config)
        self.gate = NekoMindMoe2TopKRouter(config)
        self.shared_expert = NekoMindMoe2MLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output


class NekoMindMoe2RMSNorm(LlamaRMSNorm):
    pass


class NekoMindMoe2RMSNormGated(Glm5NextTextRMSNormGated):
    pass


class NekoMindMoe2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: NekoMindMoe2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            # CODEPATH: TODO: remove this once the mlinter rule is relaxed
            NekoMindMoe2Attention(config, layer_idx)
            if config.layer_types[layer_idx] == "full_attention"
            else NekoMindMoe2DeltaAttention(config, layer_idx)
        )

        self.mlp = NekoMindMoe2SparseMoeBlock(config) if config.mlp_layer_types[layer_idx] == "sparse" else NekoMindMoe2MLP(config)

        self.input_layernorm = NekoMindMoe2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = NekoMindMoe2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.block_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        # Self attn
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class NekoMindMoe2PreTrainedModel(PreTrainedModel):
    config: NekoMindMoe2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NekoMindMoe2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _keys_to_ignore_on_load_unexpected = None
    _can_record_outputs = {
        "router_logits": OutputRecorder(NekoMindMoe2TopKRouter, index=0),
        "hidden_states": NekoMindMoe2DecoderLayer,
        "attentions": NekoMindMoe2Attention,
    }
    _is_stateful = True
    _can_compile_fullgraph = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, NekoMindMoe2ForgetGate):  # following FLA initialization
            # A_log
            init.copy_(module.A_log, init.uniform_(module.A_log, a=1.0, b=16.0).log())
            # dt_bias
            init.uniform_(module.dt_bias, a=math.log(1e-3), b=math.log(1e-1))
            dt = module.dt_bias.exp().clamp_min(1e-4)
            init.copy_(module.dt_bias, dt + torch.log(-torch.expm1(-dt)))  # (stable) inverse softplus
        elif isinstance(module, NekoMindMoe2Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, NekoMindMoe2TopKRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, NekoMindMoe2RMSNormGated):
            init.ones_(module.weight)


@auto_docstring
class NekoMindMoe2Model(NekoMindMoe2PreTrainedModel):
    def __init__(self, config: NekoMindMoe2Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [NekoMindMoe2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = NekoMindMoe2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids: torch.LongTensor = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = (position_ids + past_seen_tokens).unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class NekoMindMoe2ForCausalLM(NekoMindMoe2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = NekoMindMoe2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, NekoMindMoe2ForCausalLM

        >>> model = NekoMindMoe2ForCausalLM.from_pretrained("nekocyrene/NekoMind1.5-Base")
        >>> tokenizer = AutoTokenizer.from_pretrained("nekocyrene/NekoMind1.5-Base")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "NekoMindMoe2Config",
    "NekoMindMoe2ForCausalLM",
    "NekoMindMoe2Model",
    "NekoMindMoe2PreTrainedModel",
]
