from transformers import PretrainedConfig


class MOSSv1d6VisionConfig(PretrainedConfig):
    model_type = "moss_v1d6_vision"

    def __init__(
        self,
        image_size: tuple | int = 1536,
        hidden_size: int = 1024,
        num_channels: int = 3,
        patch_size: int = 16,
        stride: int = 16,
        head_dim: int = 128,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        intermediate_size: int = 1536,
        hidden_act: str = "silu",
        rope_theta: float = 10000.0,
        qk_norm: bool = False,
        ffn_type: str = "swiglu",
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.stride = stride
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rope_theta = rope_theta
        self.qk_norm = qk_norm
        self.ffn_type = ffn_type
        self.eps = eps
        for k, v in kwargs.items():
            setattr(self, k, v)


class MOSSv1d6TextConfig(PretrainedConfig):
    model_type = "moss_v1d6_text"

    def __init__(
        self,
        vocab_size: int = 67840,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        img_token_id: int | None = None,
        max_position_embeddings: int = 8192,
        max_length: int = 4096,
        hidden_size: int = 1024,
        head_dim: int = 128,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        addition_eos_token_ls: list[int] | None = None,
        cross_attn_layers: tuple | list | None = None,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        hidden_act: str = "silu",
        partial_rope: bool = False,
        abs_pe: bool = False,
        abs_pe_max_length: int = 8192,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.img_token_id = img_token_id
        self.max_position_embeddings = max_position_embeddings
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.addition_eos_token_ls = addition_eos_token_ls
        self.cross_attn_layers = tuple(cross_attn_layers or ())
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.partial_rope = partial_rope
        self.abs_pe = abs_pe
        self.abs_pe_max_length = abs_pe_max_length
        for k, v in kwargs.items():
            setattr(self, k, v)


class MOSSv1d6Config(PretrainedConfig):
    model_type = "moss_v1d6"
    sub_configs = {
        "encoder_config": MOSSv1d6VisionConfig,
        "decoder_config": MOSSv1d6TextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        decoder_config=None,
        encoder_config=None,
        compressor_config: dict | None = None,
        linear_adapter: bool = False,
        **kwargs,
    ):
        if isinstance(encoder_config, dict):
            self.encoder_config = self.sub_configs["encoder_config"](**encoder_config)
        elif encoder_config is None:
            self.encoder_config = self.sub_configs["encoder_config"]()
        else:
            self.encoder_config = encoder_config

        if isinstance(decoder_config, dict):
            self.decoder_config = self.sub_configs["decoder_config"](**decoder_config)
        elif decoder_config is None:
            self.decoder_config = self.sub_configs["decoder_config"](**kwargs)
        else:
            self.decoder_config = decoder_config

        self.compressor_config = compressor_config or {}
        self.linear_adapter = linear_adapter
        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False):
        return self.decoder_config

    def to_dict(self):
        output = super().to_dict()
        output.update(
            {
                "encoder_config": self.encoder_config.to_dict()
                if hasattr(self.encoder_config, "to_dict")
                else self.encoder_config,
                "decoder_config": self.decoder_config.to_dict()
                if hasattr(self.decoder_config, "to_dict")
                else self.decoder_config,
                "compressor_config": self.compressor_config,
            }
        )
        return output
