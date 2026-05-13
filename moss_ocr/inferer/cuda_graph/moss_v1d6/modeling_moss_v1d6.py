import os
import types
from typing import Optional, Callable, Union
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.generation import GenerationMixin
from moss_ocr.inferer.cuda_graph.moss_v1d6.decoder import TransformerForCasual, CasualConfig
from moss_ocr.inferer.cuda_graph.moss_v1d6.encoder import VisionTransformer, VitConfig
from moss_ocr.inferer.cuda_graph.moss_v1d6.compressor import (
    CNNBasedCompressor,
    PixelShuffleCompressor,
    Identity,
    PatchMerger,
    PatchMergerNaive,
    LearnableQueriesCompressor,
    SwiGLU_Neck
)


class MOSSV1d6Config(PretrainedConfig):
    model_type = "moss_v1d6"
    sub_configs = {"encoder_config": VitConfig, "decoder_config": CasualConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            decoder_config=None,
            encoder_config=None,
            linear_adapter: bool = False,
            **kwargs
    ):

        if isinstance(encoder_config, dict):
            self.encoder_config = self.sub_configs["encoder_config"](**encoder_config)
        elif encoder_config is None:
            self.encoder_config = self.sub_configs["encoder_config"]()

        if isinstance(decoder_config, dict):
            self.decoder_config = self.sub_configs["decoder_config"](**decoder_config)
        elif decoder_config is None:
            self.decoder_config = self.sub_configs["decoder_config"](**kwargs)
        self.linear_adapter = linear_adapter
        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        output.update({
            'encoder_config': self.encoder_config if isinstance(self.encoder_config, dict) else self.encoder_config.to_dict(),
            'decoder_config': self.decoder_config if isinstance(self.decoder_config, dict) else self.decoder_config.to_dict()
        })
        return output

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return cls(**config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        return config if not return_unused_kwargs else (config, kwargs)


class MOSSv1d6VLPreTrainedModel(PreTrainedModel):
    config_class = MOSSV1d6Config
    base_model_prefix = "model"
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True


class MOSSv1d6VLModel(MOSSv1d6VLPreTrainedModel, GenerationMixin):
    TASK_PROMPT_MAP = dict(
        math="read formula from image and output in Latex formula format: \n",
        table="read table from image and output in HTML format: \n",
        text="read text from image and output in Markdown format: \n",
    )
    SUPPORT_COMPRESSOR_TYPE = (
        "identity",
        "pixel_unshuffle",
        "cnn",
        "patch_merge",
        "patch_merge_naive",
        "learnable_queries",
        "swi_glu_neck"
    )

    COMPRESSOR_TYPE_INFO_MAP = dict(
        identity=dict(module=Identity),
        pixel_unshuffle=dict(module=PixelShuffleCompressor),
        cnn=dict(module=CNNBasedCompressor),
        patch_merge=dict(module=PatchMerger),
        patch_merge_naive=dict(module=PatchMergerNaive),
        learnable_queries=dict(module=LearnableQueriesCompressor),
        swi_glu_neck=dict(module=SwiGLU_Neck)
    )

    def __init__(self, config: MOSSV1d6Config, **kwargs):
        super().__init__(config)
        self.config = config
        self.encoder = VisionTransformer._from_config(config.encoder_config)
        self.decoder = TransformerForCasual._from_config(config.decoder_config)
        self.generation_config = self.decoder.config

        if self.config.linear_adapter:
            self.vis_lm_adapter = nn.Linear(
                self.config.encoder_config.hidden_size,
                self.config.decoder_config.hidden_size
            )
        else:
            self.vis_lm_adapter = lambda x: x
        self.compressor = None
        if hasattr(self.config, "compress_type"):
            self.compressor = self.COMPRESSOR_TYPE_INFO_MAP[self.config.compress_type]["module"](config=self.config.compressor_config)

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        if self.compressor is not None and hasattr(self.compressor, "eval"):
            self.compressor.eval()
        if self.vis_lm_adapter is not None and hasattr(self.vis_lm_adapter, "eval"):
            self.vis_lm_adapter.eval()
        return self

    def get_image_features(
            self,
            pixel_values: torch.Tensor | list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hasattr(self.encoder, "forward_packing"), f"Encoder must implement forward_packing"
        if isinstance(pixel_values, list):
            H_W_ls = [i.size()[-2:] for i in pixel_values]
            assert set([i.shape[0] for i in pixel_values]) == {1}, f"each item bsz should be `1`"
            encoder_hidden_states_ls: list[torch.Tensor] = self.encoder.forward_packing(pixel_values)
            assert encoder_hidden_states_ls[0].ndim == 3 and encoder_hidden_states_ls[0].size()[0] == 1

            if self.compressor is not None:
                if hasattr(self.compressor, "forward_packing"):
                    encoder_hidden_states_ls = self.compressor.forward_packing(encoder_hidden_states_ls, H_W_ls)
                else:
                    encoder_hidden_states_ls = [
                        self.compressor(
                            encoder_hidden_states,
                            H // self.config.encoder_config.patch_size,
                            W // self.config.encoder_config.patch_size
                        ) for (H, W), encoder_hidden_states in zip(H_W_ls, encoder_hidden_states_ls)
                    ]

            bsz = len(pixel_values)
            img_seq_length_ls = [i.size()[1] for i in encoder_hidden_states_ls]
            max_seq_length = max(img_seq_length_ls)
            img_feature_hidden_size = encoder_hidden_states_ls[0].size(-1)
            encoder_hidden_states = encoder_hidden_states_ls[0].new_zeros(bsz, max_seq_length, img_feature_hidden_size)
            cross_attention_mask = encoder_hidden_states_ls[0].new_zeros(bsz, max_seq_length)

            for bsz_id, (seq_len, cur_encoder_hidden_states) in enumerate(zip(img_seq_length_ls, encoder_hidden_states_ls)):
                cross_attention_mask[bsz_id, :seq_len] = 1
                encoder_hidden_states[bsz_id: bsz_id + 1, :seq_len, :] = cur_encoder_hidden_states

            encoder_hidden_states = self.vis_lm_adapter(encoder_hidden_states)
            return encoder_hidden_states, cross_attention_mask

        elif isinstance(pixel_values, torch.Tensor):
            encoder_hidden_states = self.encoder(pixel_values)
            if self.compressor is not None:
                H, W = pixel_values.size()[-2:]
                encoder_hidden_states = self.compressor(
                    encoder_hidden_states,
                    H // self.config.encoder_config.patch_size,
                    W // self.config.encoder_config.patch_size
                )
            cross_attention_mask = encoder_hidden_states.new_ones(*encoder_hidden_states.size()[:2])
            encoder_hidden_states = self.vis_lm_adapter(encoder_hidden_states)
            return encoder_hidden_states, cross_attention_mask

        else:
            raise NotImplementedError(f"Unsupported pixel_values type: {type(pixel_values)}")

    def forward(
            self,
            pixel_values: torch.Tensor | list[torch.Tensor] | None = None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            output_cross_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            **kwargs
    ):

        if pixel_values is not None:
            if encoder_hidden_states is None:
                encoder_hidden_states, cross_attention_mask = self.get_image_features(pixel_values)
        return self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cross_attention_mask=cross_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_cross_attentions=output_cross_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            encoder_hidden_states=encoder_hidden_states
        )

    @torch.inference_mode()
    def prepare_inputs(
            self,
            pixel_values: torch.Tensor | list[torch.Tensor],
            tasks: str | list[str],
            tokenizer: PreTrainedTokenizer
    ):
        if isinstance(tasks, str):
            tasks = [tasks]
        assert isinstance(tasks, list)
        batch_input_ids = []
        batch_attention_masks = []
        max_length = 0
        for cur_task in tasks:
            cur_prompt = tokenizer.bos_token + self.TASK_PROMPT_MAP[cur_task]
            cur_input_ids = tokenizer(cur_prompt, add_special_tokens=False)["input_ids"]
            batch_input_ids.append(cur_input_ids)
            max_length = max(max_length, len(cur_input_ids))

        for i in range(len(batch_input_ids)):
            cur_input_ids = batch_input_ids[i]
            delta = max_length - len(cur_input_ids)
            # ---------- left padding
            batch_attention_masks.append([0] * delta + [1] * len(cur_input_ids))  # left padding
            cur_input_ids_with_padding = [tokenizer.pad_token_id] * delta + cur_input_ids  # left padding
            # ----------
            batch_input_ids[i] = cur_input_ids_with_padding

        encoder_hidden_states, cross_attention_mask = self.get_image_features(pixel_values)
        attention_mask: torch.Tensor = torch.tensor(batch_attention_masks).to(dtype=torch.long, device=encoder_hidden_states.device)
        input_ids: torch.Tensor = torch.tensor(batch_input_ids).to(dtype=torch.long, device=encoder_hidden_states.device)
        position_ids: torch.Tensor = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill(attention_mask == 0, 0)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_mask=cross_attention_mask
        )

    @torch.inference_mode()
    def prefill(
        self,
        pixel_values: torch.Tensor | list[torch.Tensor],
        tasks: str | list[str],
        tokenizer: PreTrainedTokenizer,
        **kwargs
    ):
        inputs = self.prepare_inputs(pixel_values=pixel_values, tasks=tasks, tokenizer=tokenizer)
        return self.decoder.prefill(**inputs, **kwargs)

    @torch.inference_mode()
    def generate(
        self,
        pixel_values: torch.Tensor | list[torch.Tensor],
        tasks: str | list[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        **kwargs
    ):
        return self.generate_impl(pixel_values=pixel_values, tasks=tasks, tokenizer=tokenizer, max_length=max_length, **kwargs)

    @torch.inference_mode()
    def generate_impl(
            self,
            pixel_values: torch.Tensor | list[torch.Tensor],
            tasks: str | list[str],
            tokenizer: PreTrainedTokenizer,
            max_length: int = 2048,
            **kwargs
    ):
        inputs = self.prepare_inputs(pixel_values=pixel_values, tasks=tasks, tokenizer=tokenizer)
        return self.decoder.generate_impl4(**inputs, **kwargs, max_length=max_length)

    @torch.inference_mode()
    def generate_replay_cuda_graph(
            self,
            pixel_values: torch.Tensor | list[torch.Tensor],
            tasks: str | list[str],
            tokenizer: PreTrainedTokenizer,
            max_length: int = 2048,
            **kwargs
    ):
        inputs = self.prepare_inputs(pixel_values=pixel_values, tasks=tasks, tokenizer=tokenizer)
        assert hasattr(self.decoder, "generate_with_cuda_graph")
        return self.decoder.generate_with_cuda_graph(**inputs, **kwargs, max_length=max_length)

    @torch.inference_mode()
    def generate_replay_cuda_graph_streaming(
            self,
            pixel_values: torch.Tensor | list[torch.Tensor],
            tasks: str | list[str],
            tokenizer: PreTrainedTokenizer,
            max_length: int = 2048,
            **kwargs
    ):
        inputs = self.prepare_inputs(pixel_values=pixel_values, tasks=tasks, tokenizer=tokenizer)
        assert hasattr(self.decoder, "generate_with_cuda_graph_streaming")
        return self.decoder.generate_with_cuda_graph_streaming(
                **inputs,
                **kwargs,
                tokenizer=tokenizer,
                max_length=max_length
        )


def prepare_input_embeds(
        self,
        input_ids,
        encoder_hidden_states: torch.Tensor | None | list[torch.Tensor] | list[torch.Tensor] = None,
        img_global_prefix: torch.Tensor | None = None
):
    """
    Args:
        input_ids: (bsz, seq_len)
        encoder_hidden_states: when encoder_hidden_states is list, maybe image feature with different seqence length
        **kwargs:
    Returns:
        inputs_embeds: (bsz, seq_len, hidden_size)
    """
    inputs_embeds = self.embed_tokens(input_ids)
    bsz = input_ids.shape[0]
    img_feature_mask: torch.Tensor = input_ids == self.config.img_token_id  # B, L
    for i in range(bsz):
        inputs_embeds[i:i + 1] = inputs_embeds[i:i + 1].masked_scatter(
            img_feature_mask[i].unsqueeze(-1),
            encoder_hidden_states[i].to(dtype=inputs_embeds.dtype)
        )
    return inputs_embeds


class MOSSv1d6ImageAsPrefixVLModel(MOSSv1d6VLModel):
    def __init__(self, config: MOSSV1d6Config, **kwargs):
        super(MOSSv1d6ImageAsPrefixVLModel, self).__init__(config=config, **kwargs)
        self.decoder.model.prepare_input_embeds = types.MethodType(prepare_input_embeds, self.decoder.model)
        if self.compressor is None and getattr(self.config, "compressor_config", None) is not None:
            self.compressor = PatchMergerNaive(config=self.config.compressor_config)
            print(f"[MOSSv1d6ImageAsPrefixVLModel] config has no `compressor_type`, Using PatchMergerNaive as compressor")

    def get_image_features(
            self,
            pixel_values: torch.Tensor | list[torch.Tensor]
    ) -> list[torch.Tensor]:
        assert hasattr(self.encoder, "forward_packing"), f"Encoder must implement forward_packing"
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = [pixel_values]

        H_W_ls = [i.size()[-2:] for i in pixel_values]
        assert set([i.shape[0] for i in pixel_values]) == {1}, f"each item bsz should be `1`"
        encoder_hidden_states_ls: list[torch.Tensor] = self.encoder.forward_packing(pixel_values)
        assert encoder_hidden_states_ls[0].ndim == 3 and encoder_hidden_states_ls[0].size()[0] == 1

        if self.compressor is not None:
            if hasattr(self.compressor, "forward_packing"):
                encoder_hidden_states_ls = self.compressor.forward_packing(encoder_hidden_states_ls, H_W_ls)
            else:
                encoder_hidden_states_ls = [
                    self.compressor(
                        encoder_hidden_states,
                        H // self.config.encoder_config.patch_size,
                        W // self.config.encoder_config.patch_size
                    ) for (H, W), encoder_hidden_states in zip(H_W_ls, encoder_hidden_states_ls)
                ]

        encoder_hidden_states = [self.vis_lm_adapter(i) for i in encoder_hidden_states_ls]
        return encoder_hidden_states

    def prepare_inputs(
            self,
            pixel_values: torch.Tensor | list[torch.Tensor],
            tasks: str | list[str],
            tokenizer: PreTrainedTokenizer
    ):
        if isinstance(tasks, str):
            tasks = [tasks]
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = [pixel_values]
        assert isinstance(tasks, list)
        batch_input_ids = []
        batch_attention_masks = []
        encoder_hidden_states: list[torch.Tensor] = self.get_image_features(pixel_values)

        max_length = 0
        for cur_task, cur_img_fea in zip(tasks, encoder_hidden_states):
            cur_img_token_num = cur_img_fea.shape[1]
            cur_prompt = tokenizer.bos_token + self.TASK_PROMPT_MAP[cur_task]
            cur_input_ids = tokenizer(cur_prompt, add_special_tokens=False)["input_ids"]
            cur_input_ids = [self.config.decoder_config.img_token_id] * cur_img_token_num + cur_input_ids
            batch_input_ids.append(cur_input_ids)
            max_length = max(max_length, len(cur_input_ids))

        for i in range(len(batch_input_ids)):
            cur_input_ids = batch_input_ids[i]
            delta = max_length - len(cur_input_ids)
            # ---------- left padding
            batch_attention_masks.append([0] * delta + [1] * len(cur_input_ids))  # left padding
            cur_input_ids_with_padding = [tokenizer.pad_token_id] * delta + cur_input_ids  # left padding
            # ----------
            batch_input_ids[i] = cur_input_ids_with_padding

        attention_mask: torch.Tensor = torch.tensor(batch_attention_masks).to(
            dtype=torch.long, device=encoder_hidden_states[0].device
        )
        input_ids: torch.Tensor = torch.tensor(batch_input_ids).to(
            dtype=torch.long, device=encoder_hidden_states[0].device
        )
        position_ids: torch.Tensor = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill(attention_mask == 0, 0)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

