from transformers import PretrainedConfig, PreTrainedModel
import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
from contextlib import nullcontext
from dataclasses import dataclass


@dataclass
class GenerateOutput:
    sequences: torch.LongTensor | None | list[torch.LongTensor] = None  #
    scores: torch.FloatTensor | None = None  # (B, )
    logits: torch.FloatTensor | None = None  # (B, L, d)
    mask: torch.Tensor | None = None  # (B, L)
    attentions: tuple[tuple[torch.FloatTensor]] | None = None
    cross_attentions: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = None
    past_key_values: tuple[tuple[tuple[torch.FloatTensor]]] | None = None


class CasualConfig(PretrainedConfig):
    def __init__(
        self,
        model_path: str | None = None,
        vocab_size: int = 1000,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        img_token_id: int | None = None,
        max_position_embeddings: int = 1024,
        max_length: int = 1024,

        hidden_size: int = 512,
        head_dim: int | None = None,
        intermediate_size: int = 128,
        num_hidden_layers: int = 4,

        num_attention_heads: int = 2,
        num_key_value_heads: int = 2,  # self-attn support GQA
        addition_eos_token_ls: list[int] | None = None,

        cross_attn_num_key_value_heads: int | None = None,  # cross-attn support GQA
        initializer_range=0.02,
        use_qk_norm: bool = False,
        attn_bias: bool = False,
        attn_o_bias: bool = False,  # Attention layer, output linear layer bias
        self_attn_layers: tuple | list | None = None,  #
        cross_attn_layers: tuple | list | None = None,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        scale_embedding: bool = False,  # only for additive pos embedding

        output_attentions: bool = False,
        output_cross_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,

        torch_dtype: str | None = None,  # TODO:
        tie_word_embeddings: bool = False,
        gradient_checkpointing: bool = False,  #
        _attn_implementation: str = "sdpa",  # Now, only support sdpa method!
        sliding_window: int | None = None,
        hidden_act: str = "silu",
        no_bos_sampling: bool = False,
        partial_rope: bool = False,
        abs_pe: bool = False,
        abs_pe_max_length: int = 8192,
        **kwargs
    ):
        super().__init__()
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.img_token_id = img_token_id

        self.max_position_embeddings = max_position_embeddings
        self.max_length = max_length

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads  # support GQA

        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads

        if cross_attn_num_key_value_heads is None:
            cross_attn_num_key_value_heads = num_key_value_heads
        self.cross_attn_num_key_value_heads = cross_attn_num_key_value_heads

        self.initializer_range = initializer_range
        self.use_qk_norm = use_qk_norm
        self.attn_bias = attn_bias
        self.attn_o_bias = attn_o_bias
        if self_attn_layers is None:
            self_attn_layers = tuple(range(self.num_hidden_layers))
        self.self_attn_layers = self_attn_layers
        self.addition_eos_token_ls = addition_eos_token_ls

        if cross_attn_layers is None:
            cross_attn_layers = tuple()
        self.cross_attn_layers = cross_attn_layers

        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

        self.scale_embedding = scale_embedding

        self.output_attentions = output_attentions
        self.output_cross_attentions = output_cross_attentions
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict

        self.torch_dtype = torch_dtype
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        self._attn_implementation = _attn_implementation  # TODO
        self.sliding_window = sliding_window  # TODO
        self.hidden_act = hidden_act  # TODO, not used Now
        self.no_bos_sampling = no_bos_sampling
        self.kwargs = kwargs

        self.partial_rope = partial_rope
        self.abs_pe = abs_pe
        self.abs_pe_max_length = abs_pe_max_length

        for k, v in self.kwargs.items():
            setattr(self, k, v)


class RMSNorm(nn.Module):
    """
    https://arxiv.org/abs/1910.07467
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def naive_impl(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (self.dim,), self.weight, self.eps)
        else:
            return self.naive_impl(x)


def repeat_kv(keys: torch.Tensor, values, repeats: int, dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    assert dim == 1, f"Not implement other dim now. dim must be 1, got {dim}."
    if repeats == 1:
        return keys, values

    def repeat_single(_tensor: torch.Tensor, _repeats: int, _dim: int = 1) -> torch.Tensor:
        bsz, n_heads, seq_len, head_dim = _tensor.shape
        return _tensor.unsqueeze(dim=dim+1).expand(-1, -1, _repeats, -1, -1).reshape(bsz, -1, seq_len, head_dim)

    return repeat_single(keys, repeats, dim), repeat_single(values, repeats, dim)


class RotaryEmbedding(nn.Module):
    """https://arxiv.org/abs/2104.09864"""
    def __init__(
            self,
            dim: int,  # head_dim
            base: float | int = 10000,  # rope theta
            device: torch.device | None | str = None,
            dtype: torch.dtype | None = None,
            partial_rope: bool = False
    ):
        super().__init__()
        self.partial_rope = partial_rope
        if not self.partial_rope:
            self.dim = dim
        else:
            self.dim = dim // 2
        self.base = base
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        if self.device is not None:
            inv_freq = inv_freq.to(device=device, dtype=dtype)
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Reference from transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding.forward with Gemma
        """
        Args:
            position_ids: shape like (batch_size, sequence_length)
        Returns:
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """https://arxiv.org/pdf/2002.05202"""
    def __init__(self, config: CasualConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        up_states = self.gate_up_proj(x)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * F.silu(gate)
        down_proj = self.down_proj(up_states)
        return down_proj


class SdpaSelfAttention(nn.Module):
    """scale dot-product attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            config: CasualConfig,
            layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.q_size = config.num_attention_heads * self.head_dim
        self.kv_size = config.num_key_value_heads * self.head_dim
        self.use_qk_norm = getattr(self.config, "use_qk_norm", False)
        attn_bias = getattr(self.config, "attn_bias", True)
        attn_o_bias = getattr(self.config, "attn_o_bias", False)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(dim=config.head_dim)
            self.k_norm = RMSNorm(dim=config.head_dim)

        self.qkv_proj = nn.Linear(
            config.hidden_size, 
            self.q_size + self.kv_size * 2, 
            bias=attn_bias
        )
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=attn_o_bias)
        self.key_caches = None
        self.value_caches = None

    def set_static_cache(self, max_batch_size, max_length, **kwargs):
        self.key_caches = next(self.parameters()).data.new_empty(
            (max_batch_size, self.num_attention_heads, max_length, self.head_dim))
        self.value_caches = next(self.parameters()).data.new_empty(
            (max_batch_size, self.num_attention_heads, max_length, self.head_dim))

    def _apply_rope(self, position_embeddings, query_states, key_states):
        cos, sin = position_embeddings
        if not self.config.partial_rope:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:  # partial rope
            assert self.head_dim % 2 == 0
            q_nope, q_pe = torch.split(query_states, self.head_dim // 2, dim=-1)
            k_nope, k_pe = torch.split(key_states, self.head_dim // 2, dim=-1)
            q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
            query_states = torch.cat([q_nope, q_pe], dim=-1)
            key_states = torch.cat([k_nope, k_pe], dim=-1)
        return query_states, key_states

    def forward_prefill(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert self.key_caches is not None and self.value_caches is not None, f"please init static cache before forward"
        bsz, seq_len = hidden_states.shape[:2]

        hidden_shape = (bsz, seq_len, -1, self.head_dim)  # B, S, nh, head_dim
        query_states, key_states, value_states = self.qkv_proj(hidden_states).split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        query_states = query_states.view(hidden_shape).transpose(1, 2)  # B, nh, L, head_dim
        key_states = key_states.view(hidden_shape).transpose(1, 2)      # B, nh, S, head_dim
        value_states = value_states.view(hidden_shape).transpose(1, 2)  # B, nh, S, head_dim
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states, key_states = self._apply_rope(position_embeddings, query_states, key_states)
        key_states, value_states = repeat_kv(key_states, value_states, self.num_key_value_groups)

        self.key_caches[:bsz, :, :seq_len] = key_states  # F.scaled_dot_product_attention process GQA automatically
        self.value_caches[:bsz, :, :seq_len] = value_states

        # as contiguous
        query_states = query_states.contiguous()  # B, nh, L, head_dim
        key_states = key_states.contiguous()      # B, nh, S, head_dim
        value_states = value_states.contiguous()  # B, nh, S, head_dim

        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attention_mask,
            is_causal=False,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            scale=self.scaling,
            enable_gqa=False
        )

        attn_output = attn_output.transpose(1, 2).flatten(-2)
        attn_output = self.o_proj(attn_output)
        return attn_output

    def forward_generation(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor,
            cached_position: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = hidden_states.shape[:2]
        assert seq_len == 1, "forward_generation should only process one token at a time"
        input_shape = hidden_states.shape[:-1]  # B, S, dim
        hidden_shape = (*input_shape, -1, self.head_dim)  # B, S, nh, head_dim

        query_states, key_states, value_states = self.qkv_proj(hidden_states).split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        query_states = query_states.view(hidden_shape).transpose(1, 2)  # B, nh, L, head_dim
        key_states = key_states.view(hidden_shape).transpose(1, 2)  # B, nh, S, head_dim
        value_states = value_states.view(hidden_shape).transpose(1, 2)  # B, nh, S, head_dim
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states, key_states = self._apply_rope(position_embeddings, query_states, key_states)
        key_states, value_states = repeat_kv(key_states, value_states, self.num_key_value_groups)

        self.key_caches[:bsz, ...].index_copy_(
            2,  
            cached_position,  
            key_states
        )
        self.value_caches[:bsz, ...].index_copy_(
            2,
            cached_position,
            value_states
        )

        full_key_cache = self.key_caches[:bsz, ...]
        full_value_cache = self.value_caches[:bsz, ...]
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=full_key_cache,
            value=full_value_cache,
            attn_mask=attention_mask[:bsz, ...],
            is_causal=False,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            scale=self.scaling,
            enable_gqa=False
        )

        attn_output = attn_output.transpose(1, 2).flatten(-2)
        attn_output: torch.Tensor = self.o_proj(attn_output)
        return attn_output


class SdpaCrossAttention(nn.Module):
    """scale dot-product attention from 'Attention Is All You Need' paper"""
    # DO NOT consider cross-attention Cache Now!

    def __init__(self, config: CasualConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.cross_attn_num_key_value_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_attention_heads)
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

        self.use_qk_norm = getattr(self.config, "use_qk_norm", False)
        attn_bias = getattr(self.config, "attn_bias", False)
        attn_o_bias = getattr(self.config, "attn_o_bias", False)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(dim=self.num_attention_heads * self.head_dim)
            self.k_norm = RMSNorm(dim=self.num_key_value_heads * self.head_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=attn_o_bias)

        self.key_caches = None
        self.value_caches = None
        self.cached_len = None

    def set_static_cache(self, max_batch_size, max_img_length=4096, **kwargs):
        self.key_caches = next(self.parameters()).data.new_empty(
            (max_batch_size, self.num_attention_heads, max_img_length, self.head_dim))
        self.value_caches = next(self.parameters()).data.new_empty(
            (max_batch_size, self.num_attention_heads, max_img_length, self.head_dim))

    def forward_prefill(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,  #
    ) -> torch.Tensor:
        bsz, img_seq_len = encoder_hidden_states.shape[:2]

        input_shape = hidden_states.shape[:-1]  # B, L, dim
        encoder_input_shape = encoder_hidden_states.shape[:-1]  # B, S, dim
        hidden_shape = (*input_shape, -1, self.head_dim)  # B, L, nh, head_dim
        kv_hidden_shape = (*encoder_input_shape, -1, self.head_dim)  # B, S, nh, head_dim

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(encoder_hidden_states)  # B, img_seq_len, dim
        value_states = self.v_proj(encoder_hidden_states)  # B, img_seq_len, dim

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)  # B, nh, S, head_dim
        key_states = key_states.view(kv_hidden_shape).transpose(1, 2)  # B, nh, img_seq_len, head_dim
        value_states = value_states.view(kv_hidden_shape).transpose(1, 2)
        key_states, value_states = repeat_kv(key_states, value_states, self.num_key_value_groups)

        self.key_caches[:bsz, :, :img_seq_len] = key_states
        self.value_caches[:bsz, :, :img_seq_len] = value_states
        self.cached_len = img_seq_len

        # as continue
        query_states = query_states.contiguous()  # B, nh, L, head_dim
        key_states = key_states.contiguous()      # B, nh, S, head_dim
        value_states = value_states.contiguous()  # B, nh, S, head_dim

        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            is_causal=False,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            scale=self.scaling,
            enable_gqa=False
        )
        attn_output = attn_output.transpose(1, 2).flatten(-2)
        attn_output = self.o_proj(attn_output)
        return attn_output

    def forward_generation(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,  #
            **kwargs
    ) -> torch.Tensor:
        bsz = hidden_states.size(0)

        input_shape = hidden_states.shape[:-1]  # B, L, dim
        hidden_shape = (*input_shape, -1, self.head_dim)  # B, L, nh, head_dim
        query_states = self.q_proj(hidden_states)
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)  # B, nh, S, head_dim
        key_states = self.key_caches[:bsz, ...]
        value_states = self.value_caches[:bsz, ...]

        # as continue
        query_states = query_states.contiguous()  # B, nh, L, head_dim
        key_states = key_states.contiguous()      # B, nh, S, head_dim
        value_states = value_states.contiguous()  # B, nh, S, head_dim

        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attention_mask[:bsz, ...],  # No mask needed for Q_len=1
            is_causal=False,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            scale=self.scaling,
            enable_gqa=False
        )
        attn_output = attn_output.transpose(1, 2).flatten(-2)
        attn_output = self.o_proj(attn_output)
        return attn_output
        

class DecoderLayer(nn.Module):
    def __init__(self, config: CasualConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.self_attn = None
        if layer_idx in getattr(self.config, "self_attn_layers", (layer_idx, )):  # default using self_attn
            self.self_attn = SdpaSelfAttention(config=config, layer_idx=layer_idx)

        self.cross_attn = None
        if layer_idx in getattr(self.config, "cross_attn_layers", []):  # default do not use self_attn
            self.cross_attn = SdpaCrossAttention(config=config, layer_idx=layer_idx)
            self.cross_attn_input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = SwiGLU(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def set_static_cache(self, max_batch_size, max_length, max_img_length, **kwargs):
        if self.self_attn is not None:
            self.self_attn.set_static_cache(max_length=max_length, max_batch_size=max_batch_size)
        if self.cross_attn is not None:
            self.cross_attn.set_static_cache(max_batch_size=max_batch_size, max_img_length=max_img_length)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # necessary, but kept here for BC
        encoder_hidden_states: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states

        # Self Attention
        if self.self_attn is not None:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn.forward_prefill(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
            hidden_states = residual + hidden_states
            residual = hidden_states

        if self.cross_attn is not None:
            # Do cross-attention on encoder outputs
            hidden_states = self.cross_attn_input_layernorm(hidden_states)
            cross_attn_output = self.cross_attn.forward_prefill(
                hidden_states=hidden_states,
                attention_mask=cross_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = cross_attn_output + residual
            residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_generation(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # necessary, but kept here for BC
        cached_position: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states

        # Self Attention
        if self.self_attn is not None:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn.forward_generation(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                cached_position=cached_position,
            )
            hidden_states = residual + hidden_states
            residual = hidden_states

        if self.cross_attn is not None:
            # Do cross-attention on encoder outputs
            hidden_states = self.cross_attn_input_layernorm(hidden_states)
            cross_attn_output = self.cross_attn.forward_generation(
                hidden_states=hidden_states,
                attention_mask=cross_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = cross_attn_output + residual
            residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DecoderPreTrainedModel(PreTrainedModel):
    config_class = CasualConfig
    main_input_name = "input_ids"
    base_model_prefix = "model"
    _supports_sdpa = True
    supports_gradient_checkpointing = False
    _no_split_modules = ["DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class DecoderModel(DecoderPreTrainedModel):
    def __init__(self, config: CasualConfig):
        super().__init__(config=config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # , padding_idx=self.padding_idx
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.kv_cache_types = [j for i in [
            ["cross_attn_key", "cross_attn_value", "self_attn_key", "self_attn_value"]
            if layer_idx in getattr(self.config, "cross_attn_layers", [])
            else ["self_attn_key", "self_attn_value"]
            for layer_idx in range(config.num_hidden_layers)
        ] for j in i]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.config.abs_pe:
            self.abs_pe_emb = nn.Embedding(
                num_embeddings=self.config.abs_pe_max_length,
                embedding_dim=self.config.hidden_size,
                padding_idx=self.config.pad_token_id
            )

        self.rotary_emb = RotaryEmbedding(
            dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            base=config.rope_theta, partial_rope=config.partial_rope
        )

    def set_static_cache(self, max_batch_size, max_length, max_img_length, **kwargs):
        for cur_layer in self.layers:
            if hasattr(cur_layer, "set_static_cache"):
                cur_layer.set_static_cache(max_batch_size=max_batch_size, max_length=max_length, max_img_length=max_img_length, **kwargs)

    def prepare_input_embeds(
            self,
            input_ids,
            encoder_hidden_states: torch.Tensor | None | list[torch.Tensor] | list[torch.Tensor] = None,
            img_global_prefix: torch.Tensor | None = None
    ):
        inputs_embeds = self.embed_tokens(input_ids)
        bsz = input_ids.shape[0]
        if self.config.img_token_id is not None:
            img_feature_mask: torch.Tensor = input_ids == self.config.img_token_id  # B, L

            if img_feature_mask.sum() > 0:
                if img_global_prefix is None:
                    if isinstance(encoder_hidden_states, torch.Tensor):
                        assert encoder_hidden_states is not None, f"no image feature found, you should set `encoder_hidden_states`"
                        assert self.config.cross_attn_layers is None or len(self.config.cross_attn_layers) == 0, \
                            f"The model can be use either cross-attention or image-as-prefix architecture, but not both!"
                        assert img_feature_mask.sum() == torch.prod(torch.tensor(encoder_hidden_states.shape[:2])), \
                            (
                                f"image feature can but not fulfill placeholder, placeholder num: {img_feature_mask.sum()}, but "
                                f"image feature shape: {encoder_hidden_states.shape}")
                        # fulfill
                        inputs_embeds = inputs_embeds.masked_scatter(
                            img_feature_mask.unsqueeze(-1), encoder_hidden_states.to(dtype=inputs_embeds.dtype)
                        )
                    elif isinstance(encoder_hidden_states, list):
                        assert len(encoder_hidden_states) == bsz
                        for i in range(bsz):
                            inputs_embeds[i:i+1] = inputs_embeds[i:i+1].masked_scatter(
                                img_feature_mask[i].unsqueeze(-1),
                                encoder_hidden_states[i].to(dtype=inputs_embeds.dtype)
                            )
                    else:
                        raise NotImplementedError
                else:
                    img_global_prefix: torch.Tensor
                    assert img_feature_mask.sum() == torch.prod(torch.tensor(img_global_prefix.shape[:2])), \
                        (
                            f"image feature can but not fulfill placeholder, placeholder num: {img_feature_mask.sum()}, but "
                            f"image feature shape: {img_global_prefix.shape}")
                    inputs_embeds = inputs_embeds.masked_scatter(
                        img_feature_mask.unsqueeze(-1), img_global_prefix.to(dtype=inputs_embeds.dtype)
                    )
        return inputs_embeds

    def forward_prefill(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        encoder_hidden_states: torch.Tensor | None | list[torch.Tensor] = None,
        cross_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            input_ids (`torch.LongTensor`):  shape `(batch_size, sequence_length)`):
            attention_mask:
            cross_attention_mask (torch.Tensor | None). (bsz, L)
            position_ids:
            encoder_hidden_states:
            **kwargs:
        Returns:
        """

        inputs_embeds = self.prepare_input_embeds(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            img_global_prefix=kwargs.get("img_global_prefix", None)
        )

        causal_mask = self._update_causal_mask_prefill(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        cross_attention_mask = self._update_cross_attention_mask(
            cross_attention_mask=cross_attention_mask,
            dtype=inputs_embeds.dtype
        )
        if self.config.abs_pe:
            inputs_embeds = (inputs_embeds +
                             self.abs_pe_emb(position_ids).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        sin = sin.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            # remove gradient checkpoints
            decoder_layer: DecoderLayer
            hidden_states = decoder_layer.forward_prefill(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_embeddings=(cos, sin),
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_mask=cross_attention_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def forward_generation(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            cached_position: torch.Tensor,
            encoder_hidden_states: torch.Tensor | None = None,
            cross_attention_mask: torch.Tensor | None = None,
            **kwargs,
    ):
        """
        Args:
            input_ids (`torch.LongTensor`):  shape `(batch_size, sequence_length)`):
            attention_mask:
            cross_attention_mask (torch.Tensor | None). (bsz, L)
            position_ids:
            cached_position: torch.Tensor,
            encoder_hidden_states:
            **kwargs:
        Returns:
        """

        inputs_embeds = self.embed_tokens(input_ids)
        causal_mask = self._update_causal_mask_generation(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        cross_attention_mask = self._update_cross_attention_mask(
            cross_attention_mask=cross_attention_mask,
            dtype=inputs_embeds.dtype
        )
        if self.config.abs_pe:
            inputs_embeds = (inputs_embeds +
                             self.abs_pe_emb(position_ids).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        sin = sin.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            # remove gradient checkpoints
            decoder_layer: DecoderLayer
            hidden_states = decoder_layer.forward_generation(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_embeddings=(cos, sin),
                cached_position=cached_position,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_mask=cross_attention_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

    @staticmethod
    @torch.no_grad()
    def _update_cross_attention_mask(
            cross_attention_mask: torch.Tensor | None = None,
            dtype: torch.dtype | None = None
    ) -> torch.Tensor | None:
        """generate 4D casual mask for cross attention when `encode_hidden_states length` is not equal"""
        if cross_attention_mask is None:
            return None
        else:
            # attention mask shape (bsz, img_seq_len), 1: wo padding, 0 padding location
            # min_dtype = torch.finfo(dtype).min
            # cross_attention_mask = (1. - cross_attention_mask) * min_dtype
            # return cross_attention_mask[:, None, None, :]  # bsz, nhead, sequence_len, target_len
            return cross_attention_mask[:, None, None, :].bool()

    @staticmethod
    @torch.no_grad()
    def _update_causal_mask(
            attention_mask: torch.Tensor,
            inputs_embeds: torch.Tensor,
            past_seen_tokens: int,
            **kwargs
    ) -> torch.Tensor:
        """generate 4D casual mask for causal self-attention"""
        # --------  basic setting --------
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        bsz = inputs_embeds.shape[0]
        sequence_length = inputs_embeds.shape[1]
        target_length = sequence_length + past_seen_tokens
        # --------------------------------
        if sequence_length <= 1:
            return torch.zeros(bsz, 1, sequence_length, target_length).to(dtype=dtype, device=device)

        # casual_mask
        casual_mask = torch.ones(sequence_length, sequence_length).tril().to(dtype=dtype, device=device)
        casual_mask = F.pad(casual_mask, (past_seen_tokens, 0), value=1)  # (sequence_length, target_length)

        if attention_mask is not None:
            assert attention_mask.shape[-1] == sequence_length or attention_mask.shape[-1] == target_length
            if attention_mask.shape[-1] == sequence_length:
                attention_mask = F.pad(attention_mask, (past_seen_tokens, 0), value=1)  # bsz, target_length
            elif attention_mask.shape[-1] == target_length:
                attention_mask = attention_mask

        # 1: seen token, 0: unseen token
        casual_mask_with_attn = casual_mask[None, ...] * attention_mask[:, None, :]  # bsz, sequence_len, target_len
        return (1. - casual_mask_with_attn)[:, None, ...] * min_dtype

    @staticmethod
    @torch.no_grad()
    def _update_causal_mask_prefill(
            attention_mask: torch.Tensor,
            inputs_embeds: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        """generate 4D casual mask for causal self-attention"""
        # --------  basic setting --------
        causal_mask = attention_mask.new_ones(inputs_embeds.size(1), inputs_embeds.size(1)).tril()
        attention_mask = (causal_mask[None, ...] * attention_mask[:, None, :])[:, None, ...]  # bsz, 1, seq_len, target_len
        return attention_mask.bool()

    @staticmethod
    @torch.no_grad()
    def _update_causal_mask_generation(
            attention_mask: torch.Tensor,
            inputs_embeds: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        """generate 4D casual mask for causal self-attention"""
        # --------  basic setting --------
        past_seen_tokens = attention_mask.shape[1] - inputs_embeds.shape[1]
        causal_mask = attention_mask.new_ones(inputs_embeds.size(1), inputs_embeds.size(1)).tril()
        causal_mask = F.pad(causal_mask, (past_seen_tokens, 0), value=1)
        attention_mask = (causal_mask[None, ...] * attention_mask[:, None, :])[:, None, ...]  # bsz, 1, seq_len, target_len
        return attention_mask.bool()


class TransformerForCasual(DecoderPreTrainedModel):

    def __init__(self, config: CasualConfig):
        super().__init__(config=config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.model = DecoderModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._cuda_graphs = {}
        self._graph_vars = {}
        self._graph_bsz = []
        self.graph_pool = None
        self.max_length = 4096
        self.max_img_length = 4096
        self.max_batch_size = 8
        # Initialize weights and apply final processing

    def set_static_cache(self, max_batch_size, max_length, max_img_length=4096, **kwargs):
        self.max_img_length = max_img_length
        self.max_length = max_length
        self.max_batch_size = max_batch_size

        if hasattr(self.model, "set_static_cache"):
            self.model.set_static_cache(max_batch_size=max_batch_size, max_length=max_length, max_img_length=max_img_length, **kwargs)

    def get_all_cache(self):
        all_cached = []
        for module in self.model.modules():
            if hasattr(module, "key_caches"):
                all_cached.append(getattr(module, "key_caches"))
            if hasattr(module, "value_caches"):
                all_cached.append(getattr(module, "value_caches"))
        return all_cached

    def forward_prefill(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            position_ids: torch.LongTensor,
            encoder_hidden_states: torch.Tensor,
            cross_attention_mask: torch.Tensor,
            **kwargs
    ) -> tuple:

        hidden_states = self.model.forward_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_mask=cross_attention_mask,
            **kwargs
        )

        logits = self.lm_head(hidden_states)  # (B, L, d)
        return logits

    def forward_generation(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            cached_position: torch.Tensor,
            cross_attention_mask: torch.Tensor,
            encoder_hidden_states: torch.Tensor | None = None,
            **kwargs
    ):
        hidden_states = self.model.forward_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cached_position=cached_position,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_mask=cross_attention_mask,
            **kwargs
        )
        logits = self.lm_head(hidden_states)  # (B, L, d)
        return logits

    def init_cuda_graph(self, max_batch_size, max_length, max_img_length):
        """
        Args:
            max_batch_size:
            max_length: total length include prefix
            max_img_length:

        Returns:
        """
        device = next(self.parameters()).data.device
        dtype = next(self.parameters()).data.dtype
        self._graph_bsz = [1, 2, 4, 8] + list(range(16, max_batch_size + 1, 16))

        static_input_ids = torch.zeros((max_batch_size, 1), dtype=torch.long, device=device)
        static_cached_position = torch.zeros((1,), dtype=torch.long, device=device)
        static_position_ids = torch.zeros((max_batch_size, 1), dtype=torch.long, device=device)
        static_attention_mask = torch.zeros((max_batch_size, max_length), dtype=torch.long, device=device)
        static_cross_attention_mask = torch.zeros((max_batch_size, max_img_length), dtype=torch.long, device=self.device)
        static_logits = torch.zeros((max_batch_size, 1, self.config.vocab_size), dtype=dtype, device=device)

        for bsz in reversed(self._graph_bsz):
            # warm up
            for _ in range(2):
                _ = self.forward_generation(
                    input_ids=static_input_ids[:bsz, ...],
                    attention_mask=static_attention_mask[:bsz, ...],
                    position_ids=static_position_ids[:bsz, ...],
                    cached_position=static_cached_position,
                    cross_attention_mask=static_cross_attention_mask[:bsz, ...],
                )

            # Capture graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.graph_pool):
                static_logits[:bsz, ...] = self.forward_generation(
                    static_input_ids[:bsz, ...],
                    attention_mask=static_attention_mask[:bsz, ...],
                    position_ids=static_position_ids[:bsz, ...],
                    cached_position=static_cached_position,
                    cross_attention_mask=static_cross_attention_mask[:bsz, ...],
                )
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self._cuda_graphs[bsz] = graph
            torch.cuda.synchronize()

        self._graph_vars = {
            "static_input_ids": static_input_ids,
            "static_position_ids": static_position_ids,
            "static_cached_position": static_cached_position,
            "static_attention_mask": static_attention_mask,
            "static_logits": static_logits,
            "static_cross_attention_mask": static_cross_attention_mask,
        }

    @torch.inference_mode()
    def generate_impl4(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            position_ids: torch.LongTensor | None = None,
            max_length: int | None = None,
            encoder_hidden_states: torch.Tensor | None | list[torch.Tensor] = None,
            cross_attention_mask: torch.Tensor | None = None,
            output_score: bool = True,
            use_tqdm: bool = False,
            output_logits: bool = False,
            stop_on_any_eos: bool = False,
            **kwargs
    ):
        # Init Setting
        bsz, prompt_len = input_ids.shape
        device = input_ids.device

        max_complement_len = self.max_length - prompt_len
        max_length = min(max_length, max_complement_len) if max_length is not None else max_complement_len

        addition_eos_token_ls = self.config.addition_eos_token_ls if self.config.addition_eos_token_ls is not None else []
        eos_token_ls = [self.config.eos_token_id] + addition_eos_token_ls

        tokens = torch.ones((bsz, max_length), dtype=torch.long, device=device) * self.config.pad_token_id
        mask = torch.zeros((bsz, 1), device=device, dtype=torch.bool)
        if output_score:
            score_accu_sum = torch.zeros((bsz, 1), device=device, dtype=torch.float)
            count = torch.zeros((bsz, 1), device=device, dtype=torch.long)
        else:
            score_accu_sum = None
            count = None

        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill(attention_mask == 0, 0)

        logits = []
        
        static_attention_mask = attention_mask.new_zeros(bsz, self.max_length)
        static_attention_mask[:, :prompt_len].copy_(attention_mask)
        if cross_attention_mask is not None:
            static_cross_attention_mask = cross_attention_mask.new_zeros(bsz, self.max_img_length)
            static_cross_attention_mask[:, :cross_attention_mask.size(1)].copy_(cross_attention_mask)
        else:
            static_cross_attention_mask = None
        cached_position = torch.tensor([0], dtype=torch.long, device=static_attention_mask.device)

        # prefill 
        cur_logits = self.forward_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_mask=cross_attention_mask,
            **kwargs
        )

        with tqdm(range(max_length), total=max_length) if use_tqdm else nullcontext() as pbar:
            for i in (pbar if use_tqdm else range(max_length)):
                cur_pos = prompt_len + i
                cached_position.fill_(cur_pos)

                next_logits = cur_logits[:, -1:, :]  # B, 1, d
                next_input_ids = torch.argmax(next_logits, dim=-1)  # (B, 1)
                next_pos_ids = position_ids.max(dim=-1, keepdim=True).values + 1

                tokens[:, i:i + 1][~mask] = next_input_ids[~mask]
                if output_score:
                    count += (1 - mask.long())
                    score_accu_sum += (1 - mask.long()) * F.softmax(next_logits, dim=-1).max(dim=-1).values

                # update mask
                mask |= (next_input_ids == eos_token_ls[0])
                for cur_eos_tok in eos_token_ls[1:]:
                    mask |= (next_input_ids == cur_eos_tok)

                position_ids = next_pos_ids
                input_ids = next_input_ids
                static_attention_mask[:, cur_pos] = 1

                if output_logits:
                    logits.append(cur_logits[:, -1:, :])
                
                if stop_on_any_eos and torch.any(mask):
                    break
                elif torch.all(mask):
                    break       
                
                cur_logits = self.forward_generation(
                    input_ids=input_ids,
                    attention_mask=static_attention_mask,
                    position_ids=position_ids,
                    cached_position=cached_position,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_mask=static_cross_attention_mask,
                    **kwargs
                )

        if output_score:
            scores = score_accu_sum / count
        else:
            scores = None

        return GenerateOutput(
            sequences=tokens,
            logits=torch.cat(logits, dim=1) if output_logits else None,
            scores=scores,
            mask=count,
            past_key_values=None
        )
    
    @torch.inference_mode()
    def prefill(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            position_ids: torch.LongTensor | None = None,
            encoder_hidden_states: torch.Tensor | None | list[torch.Tensor] = None,
            cross_attention_mask: torch.Tensor | None = None,
            **kwargs
    ):
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

        prefill_logits = self.forward_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_mask=cross_attention_mask,
            **kwargs
        )
        return prefill_logits

    @torch.inference_mode()
    def generate_with_cuda_graph(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            position_ids: torch.LongTensor | None = None,
            max_length: int | None = None,  # complement length
            encoder_hidden_states: torch.Tensor | None | list[torch.Tensor] = None,
            cross_attention_mask: torch.Tensor | None = None,
            output_score: bool = True,
            use_tqdm: bool = False,
            output_logits: bool = False,
            stop_on_any_eos: bool = False,
            **kwargs
    ):
        # 1. Initialization
        bsz, prompt_len = input_ids.shape
        device = input_ids.device

        max_complement_len = self.max_length - prompt_len
        max_length = min(max_length, max_complement_len) if max_length is not None else max_complement_len
        eos_token_ls = [self.config.eos_token_id] + (self.config.addition_eos_token_ls or [])

        # Output tensors
        tokens = torch.full((bsz, max_length), self.config.pad_token_id, dtype=torch.long, device=device)
        all_logits = []

        # Generation state
        finished = torch.zeros(bsz, 1, dtype=torch.bool, device=device)
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

        score_accu_sum, count = None, None
        if output_score:
            score_accu_sum = torch.zeros((bsz, 1), device=device, dtype=torch.float)
            count = torch.zeros((bsz, 1), device=device, dtype=torch.long)

        # 3. Prefill Phase (process the prompt)
        prefill_logits = self.forward_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_mask=cross_attention_mask,
            **kwargs
        )

        # Get the first generated token
        next_logits = prefill_logits[:, -1:, :]
        next_token = torch.argmax(next_logits, dim=-1)

        if output_logits:
            all_logits.append(prefill_logits)

        graph = self._cuda_graphs[next(x for x in self._graph_bsz if x >= bsz)]
        self._graph_vars["static_attention_mask"].zero_()  # empty memory
        self._graph_vars["static_attention_mask"][:bsz, :attention_mask.size(1)].copy_(attention_mask)
        self._graph_vars["static_cross_attention_mask"].zero_()
        if cross_attention_mask is not None:
            self._graph_vars["static_cross_attention_mask"][:bsz, :cross_attention_mask.size(1)].copy_(cross_attention_mask)
            
        # 4. Decoding Loop (replaying the graph)
        iterator = tqdm(range(max_length)) if use_tqdm else range(max_length)
        for i in iterator:
            # Store the generated token
            tokens[:, i:i + 1] = torch.where(finished, self.config.pad_token_id, next_token)

            # Update score if needed
            if output_score:
                is_not_finished = ~finished
                count += is_not_finished.long()
                score_accu_sum += is_not_finished * F.softmax(next_logits, dim=-1).max(dim=-1).values

            # Check for EOS and update finish mask
            for eos_id in eos_token_ls:
                finished |= (next_token == eos_id)

            if stop_on_any_eos and torch.any(finished):
                break
            elif torch.all(finished):
                break

            # Prepare inputs for the next step
            cur_pos = prompt_len + i
            input_ids = next_token
            position_ids = position_ids[:, -1:, ...] + 1  # make sure left padding
            # cached_position = position_ids.max()

            # Copy dynamic data to static tensors
            self._graph_vars["static_input_ids"][:bsz, ...].copy_(input_ids)
            self._graph_vars["static_position_ids"][:bsz, ...].copy_(position_ids)
            self._graph_vars["static_cached_position"].fill_(cur_pos)
            self._graph_vars["static_attention_mask"][:bsz, cur_pos] = 1

            # Replay the graph
            graph.replay()

            # Get the output from the static logit tensor and compute next token
            next_logits = self._graph_vars["static_logits"][:bsz, ...]
            next_token = torch.argmax(next_logits, dim=-1)

            if output_logits:
                all_logits.append(next_logits)

        # 5. Finalize and Return
        if output_score:
            scores = score_accu_sum / torch.clamp(count, min=1)
        else:
            scores = None

        return GenerateOutput(
            sequences=tokens,
            logits=torch.cat(all_logits, dim=1) if output_logits and all_logits else None,
            scores=scores.squeeze(-1) if scores is not None else None,
        )

    @torch.inference_mode()
    def generate_with_cuda_graph_streaming(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            position_ids: torch.LongTensor | None = None,
            max_length: int | None = None,  # complement length
            encoder_hidden_states: torch.Tensor | None | list[torch.Tensor] = None,
            cross_attention_mask: torch.Tensor | None = None,
            output_score: bool = True,
            use_tqdm: bool = False,
            output_logits: bool = False,
            **kwargs
    ):
        # 1. Initialization
        bsz, prompt_len = input_ids.shape
        device = input_ids.device

        max_complement_len = self.max_length - prompt_len
        max_length = min(max_length, max_complement_len) if max_length is not None else max_complement_len
        eos_token_ls = [self.config.eos_token_id] + (self.config.addition_eos_token_ls or [])

        # Output tensors
        tokens = torch.full((bsz, max_length), self.config.pad_token_id, dtype=torch.long, device=device)
        all_logits = []

        # Generation state
        finished = torch.zeros(bsz, 1, dtype=torch.bool, device=device)
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

        score_accu_sum, count = None, None
        if output_score:
            score_accu_sum = torch.zeros((bsz, 1), device=device, dtype=torch.float)
            count = torch.zeros((bsz, 1), device=device, dtype=torch.long)

        # 3. Prefill Phase (process the prompt)
        prefill_logits = self.forward_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_mask=cross_attention_mask,
            **kwargs
        )

        # Get the first generated token
        next_logits = prefill_logits[:, -1:, :]
        next_token = torch.argmax(next_logits, dim=-1)

        if output_logits:
            all_logits.append(prefill_logits)

        graph = self._cuda_graphs[next(x for x in self._graph_bsz if x >= bsz)]
        self._graph_vars["static_attention_mask"].zero_()  # empty memory
        self._graph_vars["static_attention_mask"][:bsz, :attention_mask.size(1)].copy_(attention_mask)
        self._graph_vars["static_cross_attention_mask"].zero_()
        if cross_attention_mask is not None:
            self._graph_vars["static_cross_attention_mask"][:bsz, :cross_attention_mask.size(1)].copy_(
                cross_attention_mask)

        # 4. Decoding Loop (replaying the graph)

        pbar = tqdm(range(max_length)) if use_tqdm else range(max_length)
        for i in pbar:
            # Store the generated token
            yield next_token
            tokens[:, i:i + 1] = torch.where(finished, self.config.pad_token_id, next_token)

            # Update score if needed
            if output_score:
                is_not_finished = ~finished
                count += is_not_finished.long()
                score_accu_sum += is_not_finished * F.softmax(next_logits, dim=-1).max(dim=-1).values

            # Check for EOS and update finish mask
            for eos_id in eos_token_ls:
                finished |= (next_token == eos_id)
            if torch.all(finished):
                break

            # Prepare inputs for the next step
            cur_pos = prompt_len + i
            input_ids = next_token
            position_ids = position_ids[:, -1:, ...] + 1  # make sure left padding
            # cached_position = position_ids.max()

            # Copy dynamic data to static tensors
            self._graph_vars["static_input_ids"][:bsz, ...].copy_(input_ids)
            self._graph_vars["static_position_ids"][:bsz, ...].copy_(position_ids)
            self._graph_vars["static_cached_position"].fill_(cur_pos)
            self._graph_vars["static_attention_mask"][:bsz, cur_pos] = 1

            # Replay the graph
            graph.replay()

            # Get the output from the static logit tensor and compute next token
            next_logits = self._graph_vars["static_logits"][:bsz, ...]
            next_token = torch.argmax(next_logits, dim=-1)
