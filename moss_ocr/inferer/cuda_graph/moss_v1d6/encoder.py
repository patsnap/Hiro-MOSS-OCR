import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
import os
import sys
try:
    from xformers.ops.fmha import memory_efficient_attention  # type: ignore
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask  # type: ignore
except ImportError as e:
    print(f"xformers is not installed, if you wanna using packing tech, you should install it!")

try:
    import flash_attn
except ImportError as e:
    print(f"flash_attn is not installed, if you wanna using flash_attn, you should install it!")

curdir = os.path.dirname(__file__)
rtpath = os.path.join(curdir, "../../..")
sys.path.append(rtpath)


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


class Rotary2DEmbedding(nn.Module):
    """https://arxiv.org/abs/2104.09864"""

    def __init__(
            self,
            dim,
            height: int,
            width: int,
            theta: float = 10000.0,
            device: torch.device | None | str = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.device = device
        cos_sin_2d = self._compute_cos_sin_2d(dim=dim, height=height, width=width, theta=theta)
        if self.device is not None:
            cos_sin_2d = cos_sin_2d.to(device=device)
        self.register_buffer("cos_sin_2d", tensor=cos_sin_2d, persistent=False)

    @torch.no_grad()
    def forward(self, position_mesh: torch.Tensor) -> torch.Tensor:
        return self.cos_sin_2d[position_mesh[:, 0], position_mesh[:, 1]]

    @torch.no_grad()
    def _compute_cos_sin_2d(self, dim: int, height: int, width: int, theta: float) -> torch.Tensor:
        """
        Replicate the exact behavior of the complex version
        """
        # (dim / 2) frequency bases - 完全复制原始逻辑
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        h = torch.arange(height, device=inv_freq.device)
        w = torch.arange(width, device=inv_freq.device)

        inv_freq_h = torch.outer(h, inv_freq[::2]).float()
        inv_freq_w = torch.outer(w, inv_freq[1::2]).float()

        inv_freq_2d = torch.cat(
            [
                inv_freq_h[:, None, :].repeat(1, width, 1),
                inv_freq_w[None, :, :].repeat(height, 1, 1),
            ],
            dim=-1,
        )  # [height, width, dim//2]

        cos_2d = torch.cos(inv_freq_2d)
        sin_2d = torch.sin(inv_freq_2d)

        return torch.stack([cos_2d, sin_2d], dim=-1)


def apply_2D_rotary_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos_sin_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q: (bsz, head, seq_len, head_dim)
        k: (bsz, head, seq_len, head_dim)
        cos_sin_2d: (seq_len, head_dim//2, 2) where last dim is [cos, sin]
    Returns:
        Rotated q and k tensors
    """
    # cos_sin_2d shape: (seq_len, head_dim//2, 2)
    cos = cos_sin_2d[..., 0]  # (seq_len, head_dim//2)
    sin = cos_sin_2d[..., 1]  # (seq_len, head_dim//2)

    def expand(_tensor: torch.Tensor) -> torch.Tensor:
        # replace torch.repeat_interleave(_tensor, 2, dim=-1)
        _tensor = _tensor.unsqueeze(-1)  # -> (N, C, 1)
        _tensor = _tensor.expand(-1, -1, 2)  # -> (N, C, 2)
        _tensor = _tensor.reshape(_tensor.shape[0], -1)  # -> (N, C*2)
        return _tensor

    cos_expanded = expand(cos)
    sin_expanded = expand(sin)

    cos_expanded = cos_expanded.unsqueeze(0).unsqueeze(0)  # [None, None, :, :]  # (1, 1, seq_len, head_dim)
    sin_expanded = sin_expanded.unsqueeze(0).unsqueeze(0)  # [None, None, :, :]  # (1, 1, seq_len, head_dim)

    def apply_rotary_pos_emb(x, cos, sin):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        cos_part = cos[..., 0::2]
        sin_part = sin[..., 0::2]

        out1 = x1 * cos_part - x2 * sin_part
        out2 = x1 * sin_part + x2 * cos_part

        out = torch.zeros_like(x)
        out[..., 0::2] = out1
        out[..., 1::2] = out2
        return out

    q_rotated = apply_rotary_pos_emb(q, cos_expanded, sin_expanded)
    k_rotated = apply_rotary_pos_emb(k, cos_expanded, sin_expanded)

    return q_rotated, k_rotated


def apply_2D_rotary_emb_float_for_flash_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        cos_sin_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q: (seq_len, head, head_dim)
        k: (seq_len, head, head_dim)
        cos_sin_2d: (seq_len, head_dim//2, 2) where last dim is [cos, sin]
    Returns:
        Rotated q and k tensors
    """
    # cos_sin_2d shape: (seq_len, head_dim//2, 2)
    cos = cos_sin_2d[..., 0]  # (seq_len, head_dim//2)
    sin = cos_sin_2d[..., 1]  # (seq_len, head_dim//2)

    def expand(_tensor: torch.Tensor) -> torch.Tensor:
        # replace torch.repeat_interleave(_tensor, 2, dim=-1)
        _tensor = _tensor.unsqueeze(-1)  # -> (N, C, 1)
        _tensor = _tensor.expand(-1, -1, 2)  # -> (N, C, 2)
        _tensor = _tensor.reshape(_tensor.shape[0], -1)  # -> (N, C*2)
        return _tensor

    cos_expanded = expand(cos).unsqueeze(1)  # (seq_len, 1, head_dim)
    sin_expanded = expand(sin).unsqueeze(1)  # (seq_len, 1, head_dim)

    # 复现复数乘法：(a + bi) * (cos + sin*i) = (a*cos - b*sin) + (a*sin + b*cos)i
    def apply_rotary_pos_emb(x, cos, sin):
        # 将x重新整理为复数对的形式
        x1 = x[..., 0::2]  # 实部 (seq_len, num_heads, head_dim)
        x2 = x[..., 1::2]  # 虚部 (seq_len, num_heads, head_dim)

        # 对应复数乘法
        cos_part = cos[..., 0::2]  # 取对应的cos值 (seq_len, 1, head_dim)
        sin_part = sin[..., 0::2]  # 取对应的sin值 (seq_len, 1, head_dim)

        # 复数乘法公式
        out1 = x1 * cos_part - x2 * sin_part  # 新实部
        out2 = x1 * sin_part + x2 * cos_part  # 新虚部

        # 重新交错排列
        out = torch.zeros_like(x)
        out[..., 0::2] = out1
        out[..., 1::2] = out2
        return out

    q_rotated = apply_rotary_pos_emb(q, cos_expanded, sin_expanded)
    k_rotated = apply_rotary_pos_emb(k, cos_expanded, sin_expanded)

    return q_rotated, k_rotated


def repeat_kv_for_flash_attn(keys: torch.Tensor, values, repeats: int, dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    assert dim == 1, f"Not implement other dim now. dim must be 1, got {dim}."
    if repeats == 1:
        return keys, values
    
    def repeat_single(_tensor: torch.Tensor, _repeats: int, _dim: int = 1) -> torch.Tensor:
        seq_len, n_heads, head_dim = _tensor.shape
        return _tensor.unsqueeze(dim=dim+1).expand(-1, -1, _repeats, -1).reshape(seq_len, -1, head_dim)
    return repeat_single(keys, repeats, dim), repeat_single(values, repeats, dim)


class SwiGLU(nn.Module):
    """https://arxiv.org/pdf/2002.05202"""
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size * 2,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        x = F.silu(x[..., :self.intermediate_size]) * x[..., self.intermediate_size:]
        x = self.down_proj(x)
        return x


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


FFN_CLASS_MAP = {
    "swiglu": SwiGLU,
    "ffn": FFN
}


def repeat_kv(
        keys: torch.Tensor,
        values: torch.Tensor,
        repeats: int,
        dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def position_meshgrid(patch_embedding: torch.Tensor) -> torch.Tensor:
    """
    Example: input shape (1, 1, 3, 2)
    Output:
        tensor([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [2, 0],
                [2, 1]])
    Args:
        patch_embedding: shape like (B, N, H, W)
    Returns:
        mash: shape like (HxW, 2)
    """
    return torch.stack(
        torch.meshgrid(
            torch.arange(patch_embedding.shape[-2]),
            torch.arange(patch_embedding.shape[-1]),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2).to(patch_embedding.device)


def position_meshgrid_packing(
    patch_embeds_list: list[torch.Tensor],
) -> torch.Tensor:
    positions = torch.cat(
        [
            position_meshgrid(p)
            for p in patch_embeds_list
        ]
    )
    return positions


class VitConfig(PretrainedConfig):

    def __init__(
            self,
            image_size: tuple | int = 1024,
            hidden_size: int = 1024,
            num_channels: int = 3,
            patch_size: int = 16,
            stride: int = 16,
            rope_theta: float = 10000.0,  # position embedding is ROPE
            intermediate_size: int = 4096,
            num_hidden_layers: int = 24,
            num_attention_heads: int = 16,
            num_key_value_heads: int = 16,
            eps: float = 1e-6,
            qk_norm: bool = False,  # https://arxiv.org/abs/2010.04245
            model_path: str | None = None,
            gradient_checkpointing: bool = False,
            ffn_type: str | None = "swiglu",
            hidden_act: str = "silu",
            partial_rope: bool = False,  # TODO
            use_pre_norm: bool = True,
            use_post_norm: bool = False,
            **kwargs
    ):
        super().__init__()
        self.model_path = model_path
        self.gradient_checkpointing = gradient_checkpointing

        self.image_size: tuple | list = image_size if isinstance(image_size, tuple | list) else (image_size, image_size)
        self.height, self.width = self.image_size

        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.stride = stride

        self.grid_size = (
            (self.height - self.patch_size + self.stride) // self.stride,
            (self.width - self.patch_size + self.stride) // self.stride
        )
        self.rope_theta = rope_theta
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.qk_norm = qk_norm
        self.ffn_type = ffn_type
        self.hidden_act = hidden_act
        self.partial_rope = partial_rope
        self.use_pre_norm = use_pre_norm
        self.use_post_norm = use_post_norm
        self.eps = eps
        for k, v in kwargs.items():
            setattr(self, k, v)


class Attention(nn.Module):
    def __init__(self, config: VitConfig):
        super().__init__()

        self.num_attention_heads: int = config.num_attention_heads
        self.head_dim: int = config.head_dim
        self.num_key_value_heads: int = config.num_key_value_heads
        self.repeats = self.num_attention_heads // self.num_key_value_heads
        self.scale = self.head_dim ** -0.5
        self.do_qk_norm = config.qk_norm

        if config.qk_norm:
            self.q_norm = RMSNorm(dim=config.head_dim)
            self.k_norm = RMSNorm(dim=config.head_dim)

        self.qkv = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim + config.num_key_value_heads * config.head_dim + config.num_key_value_heads * config.head_dim,
            bias=False
        )
        self.q_size = config.num_attention_heads * config.head_dim
        self.kv_size = config.num_key_value_heads * config.head_dim
        self.wo = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)


    def forward(
            self,
            x: torch.Tensor,
            pos_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: shape (B, seq_len, d)
            pos_embed
        Returns:
            torch.Tensor (B, seq_len, d)
        """
        bsz, seq_len, _ = x.size()
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)  # bsz, head, seq_len, head_dim
        k = k.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        if self.do_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # apply 2D rope
        q, k = apply_2D_rotary_emb(q, k, pos_embed)
        # Repeat keys and values to match number of query heads
        k, v = repeat_kv(k, v, self.repeats, dim=1)

        # to continue
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        x = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # bsz, head, seq_len, head_dim

        # # naive impl
        # q = q * self.scale
        # attn = q @ k.transpose(-2, -1)
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = attn @ v
        x = x.transpose(1, 2).flatten(-2)  # bsz, seq_len, dim
        return self.wo(x)

    def forward_packing(
            self,
            x: torch.Tensor,
            pos_embed: torch.Tensor,
            mask: "BlockDiagonalMask"
    ) -> torch.Tensor:
        """
       Args:
           x: shape (1, seq_len, d)
           pos_embed
           mask
       Returns:
           torch.Tensor (1, seq_len, d)
       """
        bsz, seq_len, _ = x.size()
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)  # bsz, head, seq_len, head_dim
        k = k.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if self.do_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # apply 2D rope
        q, k = apply_2D_rotary_emb(q, k, pos_embed)
        # Repeat keys and values to match number of query heads
        k, v = repeat_kv(k, v, self.repeats, dim=1)

        # convert to (bsz, seq_len, head_num, head_dim)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # xformers requires (B=1, S, H, D)
        out = memory_efficient_attention(q, k, v, mask)  #
        out = out.flatten(-2)  # 1, seq_len, dim
        return self.wo(out)


    def forward_packing_flash_attn(
            self,
            x: torch.Tensor,
            pos_embed: torch.Tensor,
            **flash_attn_kwargs,
    ) -> torch.Tensor:
        seq_len, _dim = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        q = q.reshape(seq_len, self.num_attention_heads, self.head_dim)
        k = k.reshape(seq_len, self.num_key_value_heads, self.head_dim)
        v = v.reshape(seq_len, self.num_key_value_heads, self.head_dim)
        if self.do_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = apply_2D_rotary_emb_float_for_flash_attn(q, k, pos_embed)
        k, v = repeat_kv_for_flash_attn(k, v, self.repeats, dim=1)
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        ori_dtype = q.dtype
        if ori_dtype not in (torch.bfloat16, torch.float16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        return self.wo(flash_attn.flash_attn_varlen_func(q, k, v, **flash_attn_kwargs).to(ori_dtype).flatten(-2))


class TransformerBlock(nn.Module):
    def __init__(self, config: VitConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = self.config.gradient_checkpointing
        self.attention = Attention(config=config)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.eps)
        self.feed_forward = FFN_CLASS_MAP[config.ffn_type](config)

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # pre-norm https://arxiv.org/pdf/2002.04745
        _x = self.attention_norm(x)
        if self.gradient_checkpointing and self.training:
            x = x + checkpoint(self.attention.__call__, _x, pos_embed, use_reentrant=False)
            # x = x + self.attention(_x, pos_embed=pos_embed)
        else:
            x = x + self.attention(_x, pos_embed=pos_embed)
        _x = self.ffn_norm(x)
        x = x + self.feed_forward(_x)
        return x

    def forward_packing(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        mask: "BlockDiagonalMask",
        **kwargs,
    ) -> torch.Tensor:
        # pre-norm https://arxiv.org/pdf/2002.04745
        x = x + self.attention.forward_packing(self.attention_norm(x), pos_embed=pos_embed, mask=mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def forward_packing_flash_attn(
        self, 
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        **flash_attn_kwargs,
    ) -> torch.Tensor:
        x = x + self.attention.forward_packing_flash_attn(self.attention_norm(x), pos_embed=pos_embed, **flash_attn_kwargs)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class VisionTransformerBlocks(nn.Module):
    def __init__(self, config: VitConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = getattr(self.config, "gradient_checkpointing", False)
        self.layers = torch.nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(TransformerBlock(config=config))

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed)
        return x

    def forward_packing(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        mask: "BlockDiagonalMask"
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer.forward_packing(x, pos_embed=pos_embed, mask=mask)
        return x
    
    def forward_packing_flash_attn(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        **flash_attn_kwargs,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer.forward_packing_flash_attn(x, pos_embed=pos_embed, **flash_attn_kwargs)
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            config: VitConfig
    ):
        super().__init__()
        self.config = config
        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.stride,
            bias=False,
        )
        self.rope_pos_embedding = Rotary2DEmbedding(
            dim=config.head_dim,
            height=config.grid_size[0],
            width=config.grid_size[1],
            theta=config.rope_theta
        )
        self.ln_pre = RMSNorm(config.hidden_size, eps=config.eps)
        # 可选的输出层归一化，在 flash_attn 打包路径中会用到
        if self.config.use_post_norm:       
            self.ln_post = RMSNorm(config.hidden_size, eps=config.eps)
        self.transformer = VisionTransformerBlocks(config)

        if self.config.model_path is not None:
            raise NotImplementedError

    @classmethod
    def _from_config(cls, config: VitConfig) -> "VisionTransformer":
        if isinstance(config, dict):
            return cls.from_dict(config)
        return cls(config)

    @classmethod
    def from_dict(cls, config: dict) -> "VisionTransformer":
        vit_config = VitConfig.from_dict(config)
        return cls(vit_config)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:  (bsz, C, H, W)
        Returns:
            image_features: tensor of token features for all tokens of all images of
                shape (N_toks, D)
        """
        # # pass images through initial convolution independently
        # x = self.patch_conv(x)  # bsz, C, H // patch_size, W // patch_size
        # pos_mesh = position_meshgrid(x)  # positional embeddings
        # pos_embed = self.rope_pos_embedding(pos_mesh).to(device=x.device)  # torch.complex
        # if self.config.use_pre_norm:
        #     x = self.ln_pre(x.flatten(2).transpose(1, 2).contiguous())  # (bsz, seq_len, dim)
        # out = self.transformer(x, pos_embed=pos_embed)
        # if self.config.use_post_norm:
        #     x = self.ln_post(x)

        # return out  # type: ignore[no-any-return]
        return self.forward_packing([x])[0]

    def forward_packing_naive(
            self,
            x_ls: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        return [self(i) for i in x_ls]

    def _forward_packing(
            self,
            x_ls: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        https://arxiv.org/abs/2307.06304
        Args:
            x_ls: each x shape is (1, C, H, W)
        Returns:

        """
        patch_embedding_ls = [self.patch_conv(x)for x in x_ls]
        # [1, d, H // patch_size, W // patch_size]
        pos_mesh_packing = position_meshgrid_packing(patch_embedding_ls)  # positional embeddings
        packing_mask = BlockDiagonalMask.from_seqlens(
            [p.shape[-2] * p.shape[-1] for p in patch_embedding_ls]
        )
        pos_embed = self.rope_pos_embedding(pos_mesh_packing)
        # (1, all_seq_len, d)
        x = torch.cat([i.flatten(-2).transpose(-1, -2) for i in patch_embedding_ls], dim=-2).contiguous()  # packing
        if self.config.use_pre_norm:
            x = self.ln_pre(x)  # (bsz, seq_len, dim)
        out = self.transformer.forward_packing(x, pos_embed=pos_embed, mask=packing_mask)
        if self.config.use_post_norm:
            x = self.ln_post(x)
        return packing_mask.split(out)  # list of (1, seq_len, d)

    def forward_packing(self, x_ls: list[torch.Tensor]) -> list[torch.Tensor]:
        # print(f" using flash_attn packing ")
        patch_embedding_ls = [self.patch_conv(x) for x in x_ls]  # shape: list[(1, d, H // patch_size, W // patch_size)]
        pos_mesh_packing = position_meshgrid_packing(patch_embedding_ls)  # positional embeddings
        pos_embed = self.rope_pos_embedding(pos_mesh_packing)
        
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        seq_lens = []
        for patch_embedding in patch_embedding_ls:
            cur_len = patch_embedding.shape[-2] * patch_embedding.shape[-1]
            cu_seqlens_q.append(cu_seqlens_q[-1] + cur_len)
            cu_seqlens_k.append(cu_seqlens_k[-1] + cur_len)
            max_seqlen_q = max(max_seqlen_q, cur_len)
            max_seqlen_k = max(max_seqlen_k, cur_len)
            seq_lens.append(cur_len)
        
        x = torch.cat([i.flatten(-2).transpose(-1, -2) for i in patch_embedding_ls], dim=-2).contiguous().squeeze(0)  # shape: (all_seq_len, d)
        if self.config.use_pre_norm:
            x = self.ln_pre(x)  # (all_seq_len, dim)

        flash_attn_kwargs = {
            "cu_seqlens_q": torch.tensor(cu_seqlens_q).to(device=x.device, dtype=torch.int32),
            "cu_seqlens_k": torch.tensor(cu_seqlens_k).to(device=x.device, dtype=torch.int32),
            "max_seqlen_q": torch.tensor(max_seqlen_q).to(device=x.device, dtype=torch.int32),
            "max_seqlen_k": torch.tensor(max_seqlen_k).to(device=x.device, dtype=torch.int32),
            "dropout_p": 0.0,
            "causal": False,
        }
        x = self.transformer.forward_packing_flash_attn(x, pos_embed, **flash_attn_kwargs)
        if self.config.use_post_norm:
            x = self.ln_post(x)
        
        # split 
        out_ls = [i.unsqueeze(0) for i in torch.split(x, seq_lens, dim=0)]
        return out_ls
        