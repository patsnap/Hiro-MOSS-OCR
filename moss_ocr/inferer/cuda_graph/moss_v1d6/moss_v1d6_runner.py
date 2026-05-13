import os
import logging
import torch
import threading
import numpy as np
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer
from moss_ocr.inferer.cuda_graph.moss_v1d6.modeling_moss_v1d6 import MOSSv1d6VLModel, MOSSv1d6ImageAsPrefixVLModel
from moss_ocr.inferer.cuda_graph.moss_v1d6.image_processing import ImageProcess, get_nearest_image_shape
from moss_ocr.utils.utils import truncate_repetitions_fast_slice, chunk_list, to_numpy

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

curdir = os.path.dirname(os.path.abspath(__file__))


class MOSSBasicRunner(ABC):
    MODEL_PATH: str = ...
    model: MOSSv1d6VLModel | MOSSv1d6ImageAsPrefixVLModel = ...
    tokenizer: PreTrainedTokenizer = ...
    img_processor: ImageProcess = ...
    TASK_PROMPT_MAP: dict = None

    def __init__(
            self,
            model_path: str = None,
            logger=_logger,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16,
            max_batch_size: int | None = 8,
            max_length: int | None = 2048,
            max_img_length: int | None = 4096
    ):
        self.device = device
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        self.max_img_length = max_img_length
        self.model_path = model_path or self.MODEL_PATH
        self.logger = logger
        self.prepare_runner()
        self.lock = threading.Lock()
    
    @torch.inference_mode()
    def preprocessing(self, img):
        return self.img_processor(img)["pixel_values"].to(dtype=self.dtype, device=self.device)

    @torch.inference_mode()
    def img_encoder(self, pixel_values):
        H, W = pixel_values.size()[-2:]
        H = H // self.model.config.encoder_config.patch_size
        W = W // self.model.config.encoder_config.patch_size
        encoder_hidden_states = self.model.encoder(pixel_values)
        encoder_hidden_states = self.model.compressor(encoder_hidden_states, H, W)
        return encoder_hidden_states
    
    def prepare_prefill(self, img, task):
        pixel_values = self.img_processor(img)["pixel_values"].to(dtype=self.dtype, device=self.device) if isinstance(img, str) else img
        prepare_inputs = self.model.prepare_inputs(pixel_values=pixel_values, tasks=task, tokenizer=self.tokenizer)
        return prepare_inputs

    def prefill(self, prepare_inputs):
        return self.model.decoder.prefill(**prepare_inputs)

    def prepare_runner(self):
        self.logger.info(f"Start Loading model from {self.model_path}")
        self.model = self.init_model(self.model_path)
        self.tokenizer = self.init_tokenizer(self.model_path)
        self.img_processor = ImageProcess.from_pretrained(self.model_path)
        self.model = self.model.to(self.device, dtype=self.dtype)  #
        self.model = self.model.eval()
        self.model.decoder.set_static_cache(
            max_length=self.max_length,
            max_img_length=self.max_img_length,
            max_batch_size=self.max_batch_size
        )
        if self.device == "cuda":   
            self.model.decoder.init_cuda_graph(
                max_batch_size=self.max_batch_size,
                max_length=self.max_length,
                max_img_length=self.max_img_length
            )
            print(f"init cuda graph successfully! max_batch_size: {self.max_batch_size}, max_length: {self.max_length}, max_img_length: {self.max_img_length}")
        if self.TASK_PROMPT_MAP is not None:
            setattr(self.model, "TASK_PROMPT_MAP", self.TASK_PROMPT_MAP)
        self.logger.info(f"======= Finished loading model =======")
    
    def init_tokenizer(self, tokenizer_file):
        return AutoTokenizer.from_pretrained(tokenizer_file)

    @abstractmethod
    def init_model(self, model_path):
        raise NotImplementedError("Subclass must implement this method")

    @torch.inference_mode()
    def run_streaming(self, img, task, use_tqdm=True):
        pixel_values = self.img_processor(img)["pixel_values"].to(dtype=self.dtype, device=self.device)
        with self.lock:
            return self.model.generate_replay_cuda_graph_streaming(
                pixel_values=pixel_values,
                tasks=task,
                tokenizer=self.tokenizer,
                use_tqdm=use_tqdm,
                output_logits=False,
                max_length=self.max_length,
            )

    @torch.inference_mode()
    def run(self, img, task, use_tqdm=False) -> str:
        pixel_values = self.img_processor(img)["pixel_values"].to(dtype=self.dtype, device=self.device)
        with self.lock:
            res = self.model.generate_replay_cuda_graph(
                pixel_values=pixel_values,
                tasks=task,
                tokenizer=self.tokenizer,
                use_tqdm=use_tqdm,
                output_logits=False,
                max_length=self.max_length,
            )
        decoder_res = self.tokenizer.batch_decode(res.sequences, skip_special_tokens=True)
        return decoder_res[0]
    
    def _run_with_fallback(self, img, task: str, use_tqdm: bool =False, nearest_n: int = 8) -> str:
        img = to_numpy(img)
        alloc_h_w_ls = get_nearest_image_shape(img.shape[0], img.shape[1], nearest_n=nearest_n)
        pixel_values_ls = []
        for alloc_h, alloc_w in alloc_h_w_ls:
            pixel_values = self.img_processor(
                img, 
                image_shape=(alloc_h, alloc_w)
            )["pixel_values"].to(dtype=self.dtype, device=self.device)
            pixel_values_ls.append(pixel_values)
        pixel_values_chunk_ls = chunk_list(pixel_values_ls, chunk_size=self.max_batch_size) 
        cache_results = []
        for pixel_values_chunk in pixel_values_chunk_ls:
            with self.lock:
                res = self.model.generate_replay_cuda_graph(
                    pixel_values=pixel_values_chunk,
                    tasks=[task] * len(pixel_values_chunk),
                    tokenizer=self.tokenizer,
                    use_tqdm=use_tqdm,
                    output_logits=False,
                    stop_on_any_eos=True,
                    max_length=self.max_length,
                )
            cache_results.extend(res.sequences)
            # found with one has eos token
            for i in range(len(res.sequences)):
                if torch.any(res.sequences[i] == self.tokenizer.eos_token_id):
                    return self.tokenizer.decode(res.sequences[i], skip_special_tokens=True)
        return self.tokenizer.batch_decode(cache_results[-1:], skip_special_tokens=True)[0]

    def run_with_fallback(self, img, task, use_tqdm=False, nearest_n: int = 8) -> str:
        result = self.run(img, task, use_tqdm)
        truncated_result = truncate_repetitions_fast_slice(result)
        if truncated_result == result:
            return result
        return self._run_with_fallback(img=img, task=task, use_tqdm=use_tqdm, nearest_n=nearest_n)

    @torch.inference_mode()
    def run_batch(self, img_ls, task_ls, use_tqdm=False) -> list[str]:
        img_chunk_ls = [img_ls[i: i + self.max_batch_size] for i in range(0, len(img_ls), self.max_batch_size)]
        task_chunk_ls = [task_ls[i: i + self.max_batch_size] for i in range(0, len(img_ls), self.max_batch_size)]
        total_result = []
        for cur_img_ls, cur_task_ls in zip(img_chunk_ls, task_chunk_ls):
            pixel_values = [
                self.img_processor(i)["pixel_values"].to(dtype=self.dtype, device=self.device) for i in cur_img_ls
            ]
            with self.lock:
                res = self.model.generate_replay_cuda_graph(
                    pixel_values=pixel_values,
                    tasks=cur_task_ls,
                    tokenizer=self.tokenizer,
                    use_tqdm=use_tqdm,
                    output_logits=False,
                    max_length=self.max_length,
                )
            decoder_res = self.tokenizer.batch_decode(res.sequences, skip_special_tokens=True)
            total_result.extend(decoder_res)
        return total_result

    def run_batch_with_fallback(self, img_ls, task_ls, use_tqdm=False, nearest_n=8) -> list[str]:
        result = self.run_batch(img_ls, task_ls, use_tqdm)
        truncated_result = [truncate_repetitions_fast_slice(r) for r in result]
        final_result = []
        for r, t, cur_img, cur_task in zip(result, truncated_result, img_ls, task_ls):
            if t == r:
                final_result.append(r)
            else:
                final_result.append(
                    self._run_with_fallback(img=cur_img, task=cur_task, use_tqdm=use_tqdm, nearest_n=nearest_n)
                )
        return final_result


class MOSSv1d6Runner(MOSSBasicRunner):

    def init_model(self, model_path):
        return MOSSv1d6ImageAsPrefixVLModel.from_pretrained(model_path)
    