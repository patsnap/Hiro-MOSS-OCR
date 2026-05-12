import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


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
        # if hasattr(F, "rms_norm"):
        #     return F.rms_norm(x, (self.dim,), self.weight, self.eps)
        # else:
        return self.naive_impl(x)


def repeat_kv(
        keys: torch.Tensor,
        values: torch.Tensor,
        repeats: int,
        dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


class SwiGLU(nn.Module):
    """https://arxiv.org/pdf/2002.05202"""
    def __init__(self, input_channels: int, intermediate_size: int, output_channels: int | None = None):
        super().__init__()
        self.hidden_size = input_channels
        self.intermediate_size = intermediate_size
        self.output_channels = output_channels if output_channels is not None else input_channels
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.output_channels, bias=False)

    def forward(self, x):
        up_states = self.gate_up_proj(x)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * F.silu(gate)
        down_proj = self.down_proj(up_states)
        return down_proj


class LearnableQueriesCompressor(nn.Module):
    """scale dot-product attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config["input_channels"]
        self.output_channels = config["output_channels"]

        self.num_attention_heads = config.get("num_attention_heads", 8)
        self.num_key_value_heads = config.get("num_key_value_heads", 8)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.head_dim = self.config.get("head_dim", self.hidden_size // self.num_attention_heads)
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = self.config.get("attention_dropout", False)

        self.use_qk_norm = self.config.get("use_qk_norm", False)
        attn_bias = self.config.get("attn_bias", False)
        attn_o_bias = self.config.get("attn_o_bias", False)

        self.learnable_queries = nn.Parameter(
            torch.randn(1, self.config["num_learnable_queries"], self.config["input_channels"]) * 0.02
        )

        if self.use_qk_norm:
            self.q_norm = RMSNorm(dim=self.num_attention_heads * self.head_dim)
            self.k_norm = RMSNorm(dim=self.num_key_value_heads * self.head_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.output_channels, bias=attn_o_bias)

    def forward(
            self,
            x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        queries = self.learnable_queries.expand(x.size(0), -1, -1)

        hidden_shape = (*queries.shape[:-1], -1, self.head_dim)  # B, L, nh, head_dim
        kv_hidden_shape = (*x.shape[:-1], -1, self.head_dim)  # B, S, nh, head_dim
        # following key_states, value_states also can cache
        query_states = self.q_proj(queries)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)  # B, nh, S, head_dim
        key_states = key_states.view(kv_hidden_shape).transpose(1, 2)
        value_states = value_states.view(kv_hidden_shape).transpose(1, 2)

        key_states, value_states = repeat_kv(key_states, value_states, self.num_key_value_groups, dim=1)

        # as continue
        query_states = query_states.contiguous()  # B, nh, L, head_dim
        key_states = key_states.contiguous()      # B, nh, S, head_dim
        value_states = value_states.contiguous()  # B, nh, S, head_dim

        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            is_causal=False,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            scale=self.scaling
        )
        attn_output = attn_output.transpose(1, 2).flatten(-2).contiguous()  # (B, L, nh*head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output


class PatchMerger(nn.Module):
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__()
        self.hidden_size = config["input_channels"] * (config['ratio'] ** 2)
        self.layer_norm = RMSNorm(config["input_channels"], eps=1e-6)
        self.mlp = SwiGLU(
            input_channels=self.hidden_size,
            intermediate_size=config["intermediate_size"],
            output_channels=config["output_channels"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.layer_norm(x).view(-1, self.hidden_size))
        return x


class PatchMergerNaive(nn.Module):
    def __init__(self, config: dict):
        super(PatchMergerNaive, self).__init__()
        self.ratio = config['ratio']
        self.hidden_size = config["input_channels"] * (config['ratio'] ** 2)
        self.layer_norm = RMSNorm(config["input_channels"], eps=1e-6)
        self.mlp = SwiGLU(
            input_channels=self.hidden_size,
            intermediate_size=config["intermediate_size"],
            output_channels=config["output_channels"]
        )
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, D = x.shape
        x = self.layer_norm(x)
        assert L == H * W, f"Shape mismatch: {L=} != {H}x{W}"
        x = einops.rearrange(x, 'b (hr rh wr rw) d -> b (hr wr) (rh rw d)', hr=H//self.ratio, wr=W//self.ratio, rh=self.ratio, rw=self.ratio)
        return self.mlp(x)


class PixelShuffleCompressor(nn.Module):
    def __init__(self, config: dict):
        super(PixelShuffleCompressor, self).__init__()
        self.ratio = config['ratio']
        self.hidden_size = config["input_channels"] * (config['ratio'] ** 2)
        self.layer_norm = RMSNorm(config["input_channels"], eps=1e-6)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.ratio)
        self.mlp = SwiGLU(
            input_channels=self.hidden_size,
            intermediate_size=config["intermediate_size"],
            output_channels=config["output_channels"]
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, D = x.shape
        x = self.layer_norm(x)
        assert L == H * W, f"Shape mismatch: {L=} != {H}x{W}"
        x = x.transpose(1, 2).reshape(B, D, H, W).contiguous()
        x = self.pixel_unshuffle(x)  # pixel unshuffle: (B, D, H, W) -> (B, D*r*r, H/r, W/r)
        x = x.flatten(2).transpose(1, 2).contiguous()  # (B, L', D')
        x = self.mlp(x)
        return x


class PixelShuffleCompressorPacking(nn.Module):
    def __init__(self, config: dict):
        super(PixelShuffleCompressorPacking, self).__init__()
        self.ratio = config['ratio']
        self.hidden_size = config["input_channels"] * (config['ratio'] ** 2)
        self.layer_norm = RMSNorm(config["input_channels"], eps=1e-6)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.ratio)
        self.mlp = SwiGLU(
            input_channels=self.hidden_size,
            intermediate_size=config["intermediate_size"],
            output_channels=config["output_channels"]
        )

    def forward(self, x: torch.Tensor, grid_hw):
        """
        x: (L, D), L = (L1 + L2 + ... + LB), L1, L2, ..., LB may be different
        grid_hw: (B, 2)
        """
        L, D = x.shape
        seq_lens = grid_hw.prod(dim=1)
        assert L == seq_lens.sum(), f"Shape mismatch: {L=} != {seq_lens.sum()}"
        x = self.layer_norm(x)
        x = torch.cat([
            self.pixel_unshuffle(
                i.reshape(
                    grid_hw[idx][0], grid_hw[idx][1], D
                ).permute(2, 0, 1)  # (H, W, D) -> (D, H, W)
            ).flatten(1).transpose(0, 1)  # (D, H, W) -> (D * r * r, H / r, W / r) -> (D * r*r, L) -> (L, D * r * r)
            for idx, i in enumerate(torch.split(x, seq_lens.tolist(), dim=0))
        ], dim=0)
        return self.mlp(x)


class PixelShuffleCompressorPackingV2(nn.Module):
    def __init__(self, config: dict):
        super(PixelShuffleCompressorPackingV2, self).__init__()
        self.ratio = config['ratio']
        self.origin_hidden_size = config["input_channels"]
        self.hidden_size = config["input_channels"] * (config['ratio'] ** 2)
        self.layer_norm = RMSNorm(config["input_channels"], eps=1e-6)
        self.mlp = SwiGLU(
            input_channels=self.hidden_size,
            intermediate_size=config["intermediate_size"],
            output_channels=config["output_channels"]
        )

    def forward(self, x: torch.Tensor):
        """
        x: (L, D), L = (L1 + L2 + ... + LB), L1, L2, ..., LB may be different
        """

        x = x.view(-1, self.hidden_size)  
        # tile_00 = x[:, :self.origin_hidden_size]
        # tile_01 = x[:, self.origin_hidden_size:2*self.origin_hidden_size]
        # tile_10 = x[:, 2*self.origin_hidden_size:3*self.origin_hidden_size]
        # tile_11 = x[:, 3*self.origin_hidden_size:]

        # new_format_output = torch.full_like(x, fill_value=0.0)
        # new_format_output[:, 0::4] = tile_00
        # new_format_output[:, 1::4] = tile_01
        # new_format_output[:, 2::4] = tile_10
        # new_format_output[:, 3::4] = tile_11
        
        return self.mlp(x)



class CNNBasedCompressor(nn.Module):
    def __init__(self, config: dict):
        super(CNNBasedCompressor, self).__init__()
        self.modules = config["cnn_modules"]
        self.layer_norm = RMSNorm(config["input_channels"], eps=1e-6)
        self.cnn = nn.Sequential(
            *[getattr(nn, i["name"])(**i["params"]) for i in self.modules]
        )
        self.mlp = SwiGLU(
            input_channels=config["input_channels"],
            intermediate_size=config["intermediate_size"],
            output_channels=config["output_channels"]
        )

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, L, D = x.size()
        assert L == H * W, f"shape not match, {H=}, {W=}, {L=}"
        x = self.layer_norm(x)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(B, D, H, W).contiguous()
        x = self.cnn(x)
        x = x.flatten(2).contiguous()
        x = x.transpose(1, 2).contiguous()
        return x


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x


class SwiGLU_Neck(nn.Module):
    def __init__(self, config: dict):
        super(SwiGLU_Neck, self).__init__()
        self.config = config
        self.hidden_size = config["input_channels"]
        self.output_channels = config["output_channels"]
        self.intermediate_size = config["intermediate_size"]
        self.layer_norm = RMSNorm(config["input_channels"], eps=1e-6)
        self.mlp = SwiGLU(
            input_channels=self.hidden_size,
            intermediate_size=self.intermediate_size,
            output_channels=self.output_channels
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


def unite_testing_pixel_shuffle_compressor_packing():
    from einops import rearrange

    
    H_W_ls = [(16, 8), (8, 8), (4, 8)]
    merge_size = 2
    x_ls =[torch.randn(i[0], i[1], 512) for i in H_W_ls]
    x_seq_ls = [i.reshape(-1, 512) for i in x_ls]
    x_seq_premerge_seq_ls = [
        rearrange(i, '(M1 h) (M2 w) D -> (h w M1 M2) D', h=i.shape[0] // merge_size, w=i.shape[1] // merge_size, M1=merge_size, M2=merge_size) for i in x_ls
    ]
    x_seq_premerge_seq = torch.cat(x_seq_premerge_seq_ls, dim=0)
    print(x_seq_premerge_seq.shape)

    x = torch.cat(x_seq_ls, dim=0)

    compressor = PixelShuffleCompressorPacking(
        config=dict(
            ratio=2, 
            input_channels=512, 
            output_channels=512, 
            intermediate_size=512
        )
    )

    stat_dict = compressor.state_dict()
    compressor_ori = PixelShuffleCompressor(
        config=dict(
            ratio=2, 
            input_channels=512, 
            output_channels=512, 
            intermediate_size=512
        )
    )

    compressor_packing = PixelShuffleCompressorPackingV2(
        config=dict(
            ratio=2, 
            input_channels=512, 
            output_channels=512, 
            intermediate_size=512
        )
    )

    compressor_ori.eval()
    compressor_packing.eval()
    compressor.eval()

    compressor_ori.load_state_dict(stat_dict, strict=True)
    compressor_packing.load_state_dict(stat_dict, strict=True)


    ori_result_ls = [compressor_ori(x.unsqueeze(0), H_W_ls[idx][0], H_W_ls[idx][1]).squeeze(0) for idx, x in enumerate(x_seq_ls)]
    ori_result = torch.cat(ori_result_ls, dim=0)
    packing_result = compressor_packing(x_seq_premerge_seq)

    grid_hw = torch.tensor(H_W_ls, dtype=torch.long)
    x = compressor(x, grid_hw)
    print(x.shape)
    delta = x - ori_result
    delta_packing = packing_result - ori_result
    print(delta.max(), delta.min(), delta.mean())
    print(delta_packing.max(), delta_packing.min(), delta_packing.mean())

    pass


if __name__ == "__main__":
    unite_testing_pixel_shuffle_compressor_packing()
    pass