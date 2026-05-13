
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Any, Literal, TypeAlias, List, Optional, Dict
import json, os
import numpy as np
from PIL import Image
import cv2 
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BatchFeature

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    EmbeddingItems,
    ImageEmbeddingItems,
    ImageProcessorItems,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from transformers import AutoTokenizer
from vllm.transformers_utils.configs.moss_v1d5 import MOSSv1d5Config, MOSSv1d5VisionConfig

# ---------------------------------------------------------------------------
# TensorSchema helpers
# ---------------------------------------------------------------------------

class MOSSImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, TensorShape("total_seq_lens", "c", "h", "w", dynamic_dims={"h", "w"})]
    grid_hw: Annotated[torch.Tensor, TensorShape("bn", 2)]


MOSSImageInputs: TypeAlias = MOSSImagePixelInputs 


# ---------------------------------------------------------------------------
# Basic image processing
# ---------------------------------------------------------------------------

# image processing
CANDIDATE_IMAGE_SHAPE_LS = [
    (32, 1024), (32, 512), (32, 768), (32, 896), (64, 1024), (64, 512), (64, 896), (64, 768), 
    (96, 1024), (96, 768), (96, 896), (96, 512), (128, 512), (128, 768), (128, 1024), (128, 896), 
    (192, 768), (192, 512), (192, 896), (192, 1024), (256, 1024), (256, 512), (256, 768), (256, 896), 
    (320, 896), (320, 768), (320, 1024), (320, 512), (384, 512), (384, 768), (384, 1024), (384, 896), 
    (448, 512), (448, 768), (448, 1024), (448, 896), (512, 64), (512, 256), (512, 448), (512, 576), 
    (512, 384), (512, 960), (512, 128), (512, 896), (512, 32), (512, 832), (512, 704), (512, 512), 
    (512, 192), (512, 320), (512, 96), (512, 640), (512, 1024), (512, 768), (576, 896), (576, 1024), 
    (576, 768), (576, 512), (640, 896), (640, 1024), (640, 512), (640, 768), (704, 896), (704, 1024), 
    (704, 768), (704, 512), (768, 64), (768, 320), (768, 448), (768, 128), (768, 32), (768, 832), 
    (768, 704), (768, 512), (768, 192), (768, 96), (768, 896), (768, 960), (768, 768), (768, 1024), 
    (768, 576), (768, 256), (768, 384), (768, 640), (832, 512), (832, 896), (832, 768), (832, 1024), 
    (896, 512), (896, 384), (896, 320), (896, 1024), (896, 768), (896, 832), (896, 192), (896, 64), 
    (896, 32), (896, 960), (896, 96), (896, 576), (896, 896), (896, 448), (896, 256), (896, 640), 
    (896, 704), (896, 128), (960, 896), (960, 512), (960, 768), (960, 1024), (1024, 384), (1024, 32), 
    (1024, 960), (1024, 896), (1024, 128), (1024, 256), (1024, 704), (1024, 448), (1024, 576), (1024, 832), 
    (1024, 512), (1024, 96), (1024, 640), (1024, 192), (1024, 320), (1024, 768), (1024, 1024), (1024, 64)
]

CANDIDATE_IMAGE_SHAPE_ARR = np.array(CANDIDATE_IMAGE_SHAPE_LS)


def get_nearest_image_shape(height: int, width: int, nearest_n: int = 1) -> list[tuple[int, int]]:
    cur_hw = np.array([[height, width]])  # (1, 2)
    dist = np.sum(abs(cur_hw - CANDIDATE_IMAGE_SHAPE_ARR), axis=-1)  # (N, )
    alloc_h_w_ls = CANDIDATE_IMAGE_SHAPE_ARR[np.argsort(dist)][:nearest_n].tolist()
    return alloc_h_w_ls


@dataclass
class ImageProcessOutput:
    pixel_values: torch.Tensor
    grid_hw: torch.Tensor | None = None

    def __getitem__(self, key):
        return getattr(self, key)

#
class NormalizeHub:
    def __init__(self, name: str, params: Optional[Dict] = None):
        if params is None:
            params = dict()
        self.normalize_map = {
            "general_norm": self.general_normalize,
            "general_norm_v1": self.general_normalize_v1,
            "general_norm_v2": self.general_normalize_v2,
            "standard_norm": self.standard_normalize
        }
        assert name in self.normalize_map, f"{name} not in {self.normalize_map.keys()}"
        self.normalize_func = partial(self.normalize_map.get(name), **params)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.normalize_func(img)

    @staticmethod
    def general_normalize_v2(img: np.ndarray, **kwargs) -> np.ndarray:
        img /= 127.5
        return (img - 1)

    @staticmethod
    def general_normalize_v1(img: np.ndarray, **kwargs) -> np.ndarray:
        img /= 255.0
        return img

    @staticmethod
    def general_normalize(img: np.ndarray, **kwargs) -> np.ndarray:
        """normalize image to [0, 1] by ((img / 255.) - 0.5) / 0.5"""
        img /= 255.0
        img -= 0.5
        img /= 0.5
        return img

    @staticmethod
    def standard_normalize(
            img: np.ndarray,
            mean: list | tuple | None = None,
            std: list | tuple | None = None,
            **kwargs
    ) -> np.ndarray:
        """
        normalize image by ((img / 255.0) - mean) / std
        Args:
            img:  image mode is RGB
            mean: RGB's mean
            std: RGB's std

        Returns: np.ndarray
        """
        assert img.ndim == 3
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        img /= 255.0
        img -= mean
        img /= std
        return img


def to_numpy(img: np.ndarray | Image.Image | str) -> np.ndarray:
    if isinstance(img, str):
        assert os.path.exists(img)
        return cv2.imread(img)[..., ::-1]
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, Image.Image):
        return np.array(img.convert('RGB'))[..., :3]
    raise TypeError(f"Unsupported type {type(img)}")


def normalize_shape(
        height: int,
        width: int,
        min_height_size: int = 32,
        min_width_size: int = 32,
        max_height_size: int = 1344,
        max_width_size: int = 1344,
        stride: int = 16
) -> tuple[int, int]:
    """Normalize image dimensions to meet min/max constraints and align to stride."""

    if height <= 0 or width <= 0:
        return min_height_size, min_width_size

    scale_min = max(
        min_height_size / height if height < min_height_size else 0.,
        min_width_size / width if width < min_width_size else 0.
    )
    scale_max = min(
        max_height_size / height if height > max_height_size else float('inf'),
        max_width_size / width if width > max_width_size else float('inf')
    )

    scale = max(scale_min, 1.0)  # 至少保持原大小
    scale = min(scale, scale_max) if scale_max != float('inf') else scale

    if scale != 1.0:
        height = round(height * scale)
        width = round(width * scale)

    height = max(min(round(height / stride) * stride, max_height_size), min_height_size)
    width = max(min(round(width / stride) * stride, max_width_size), min_width_size)

    return height, width


def resize_with_padding_old(
        img: np.ndarray,
        image_shape: tuple[int | None, int | None] | int,
        padding_value: int = 255,
        keep_aspect_ratio: bool = True,
        center_pad: bool = True,
        no_scale_up: bool = True,
        return_crop_info: bool = False,
        interpolation: int = cv2.INTER_LINEAR,  # only influence opencv backend
        backend: str = "opencv"
) -> np.ndarray | tuple[np.ndarray, tuple]:
    assert img.ndim == 3
    if isinstance(image_shape, int):
        image_shape: tuple[int, int] = (image_shape, image_shape)
    img_dtype = img.dtype
    tgt_height, tgt_width = image_shape[:2]
    ori_h, ori_w = img.shape[:2]
    r = min(tgt_height / ori_h, tgt_width / ori_w)
    if not keep_aspect_ratio:
        if backend == "opencv":
            img = cv2.resize(img, image_shape[::-1], interpolation=interpolation)
        elif backend == "PIL":
            img = Image.fromarray(img.astype(np.uint8)).resize(image_shape[::-1], resample=2)  # default `BILINEAR`
            img = np.array(img).astype(img_dtype)
        else:
            raise NotImplementedError(f"Unsupported backend: {backend}")
        new_h, new_w = img.shape[:2]
        r_h = new_h / ori_h
        r_w = new_w / ori_w
        if not return_crop_info:
            return img
        else:
            start_x, start_y = 0, 0
            return img, (start_y, start_y + new_h, start_x, start_x + new_w, r_h, r_w)
    if no_scale_up:
        r = min(r, 1.0)
    new_h, new_w = max(math.floor(r * ori_h), 1), max(math.floor(r * ori_w), 1)
    if (new_h, new_w) == (ori_h, ori_w):
        new_img = img
    else:
        if backend == "opencv":
            new_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        elif backend == "PIL":
            new_img = Image.fromarray(img.astype(np.uint8)).resize((new_w, new_h), resample=2)  # default `BILINEAR`
            new_img = np.array(new_img).astype(img_dtype)
        else:
            raise NotImplementedError(f"Unsupported backend: {backend}")
    delta_h = tgt_height - new_img.shape[0]
    delta_w = tgt_width - new_img.shape[1]
    if center_pad:
        start_x, start_y = math.floor(delta_w / 2), math.floor(delta_h / 2)
    else:
        start_x, start_y = 0, 0
    bg = np.ones((tgt_height, tgt_width, 3), dtype=img.dtype) * padding_value
    bg[start_y: start_y + new_h, start_x: start_x + new_w, :] = new_img
    if not return_crop_info:
        return bg
    else:
        return bg, (start_y, start_y + new_h, start_x, start_x + new_w, r, r)


def resize_with_padding(
        img: np.ndarray,
        image_shape: tuple[int | None, int | None] | int,
        padding_value: int = 255,
        keep_aspect_ratio: bool = True,
        center_pad: bool = True,
        no_scale_up: bool = True,
        return_crop_info: bool = False,
        interpolation: int = cv2.INTER_LINEAR,
        backend: str = "opencv"
) -> np.ndarray | tuple[np.ndarray, tuple]:
    assert img.ndim == 3
    if isinstance(image_shape, int):
        image_shape = (image_shape, image_shape)
    
    tgt_height, tgt_width = image_shape[:2]
    ori_h, ori_w = img.shape[:2]
    r = min(tgt_height / ori_h, tgt_width / ori_w)
    img_dtype = img.dtype
    
    if not keep_aspect_ratio:
        if backend == "opencv":
            img = cv2.resize(img, image_shape[::-1], interpolation=interpolation)
        elif backend == "PIL":
            img = Image.fromarray(img.astype(np.uint8)).resize(image_shape[::-1], resample=2)  # default `BILINEAR`
            img = np.array(img).astype(img_dtype)
        else:
            raise NotImplementedError(f"Unsupported backend: {backend}")
        new_h, new_w = img.shape[:2]
        r_h = new_h / ori_h
        r_w = new_w / ori_w
        if not return_crop_info:
            return img
        else:
            start_x, start_y = 0, 0
            return img, (start_y, start_y + new_h, start_x, start_x + new_w, r_h, r_w)
        
    if no_scale_up:
        r = min(r, 1.0)
    
    new_h, new_w = max(math.floor(r * ori_h), 1), max(math.floor(r * ori_w), 1)
    
    if (new_h, new_w) == (ori_h, ori_w):
        new_img = img
    else:
        if backend == "opencv":
            new_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        elif backend == "PIL":
            new_img = Image.fromarray(img).resize((new_w, new_h), resample=2)
            new_img = np.array(new_img)
        else:
            raise NotImplementedError
            
    delta_h = tgt_height - new_h
    delta_w = tgt_width - new_w
    
    if center_pad:
        top = math.floor(delta_h / 2)
        left = math.floor(delta_w / 2)
    else:
        top, left = 0, 0
        
    bottom = delta_h - top
    right = delta_w - left

    bg = cv2.copyMakeBorder(
        new_img, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, 
        value=[padding_value, padding_value, padding_value]
    )

    if not return_crop_info:
        return bg
    else:
        return bg, (top, top + new_h, left, left + new_w, r, r)


def normalize_shape_resize_with_padding(
        img: np.ndarray,
        min_height_size: int = 32,
        min_width_size: int = 32,
        max_height_size: int = 1344,
        max_width_size: int = 1344,
        fixed_factor: int = 16,
        padding_value: int = 255,
        keep_aspect_ratio: bool = True,
        center_pad: bool = True,
        no_scale_up: bool = True,
        return_crop_info: bool = False,
        interpolation: int = cv2.INTER_LINEAR,
        backend: str = "opencv",
        image_shape: tuple[int, int] | None = None
) -> np.ndarray | tuple[np.ndarray, tuple]:
    ori_h, ori_w = img.shape[:2]
    if image_shape is not None:
        h, w = image_shape
        assert min_height_size <= h <= max_height_size and min_width_size <= w <= max_width_size, \
            f"Image shape {h}x{w} is not in the range of {min_height_size}x{min_width_size} to {max_height_size}x{max_width_size}"
        assert h % fixed_factor == 0 and w % fixed_factor == 0, f"Image shape {h}x{w} is not divisible by {fixed_factor}"
    else:
        h, w = normalize_shape(
            height=ori_h,
            width=ori_w,
            min_height_size=min_height_size,
            min_width_size=min_width_size,
            max_height_size=max_height_size,
            max_width_size=max_width_size,
            stride=fixed_factor
        )
    return resize_with_padding(
        img,
        image_shape=(h, w),
        padding_value=padding_value,
        keep_aspect_ratio=keep_aspect_ratio,
        center_pad=center_pad,
        no_scale_up=no_scale_up,
        return_crop_info=return_crop_info,
        interpolation=interpolation,
        backend=backend
    )


class ImageProcessPacking:
        
    def __init__(
            self,
            do_resize: bool = False,
            do_permute: bool = False,
            do_normalize: bool = False,
            resize_config: dict | None = None,
            normalize_config: dict | None = None,
            padding_value: int = 255,
            num_workers: int = 1,
            **kwargs
    ):
        self.do_resize = do_resize
        self.do_permute = do_permute
        self.do_normalize = do_normalize

        self.resize_config = resize_config if resize_config is not None else {}
        self.normalize_config = normalize_config if normalize_config is not None else {}

        self.padding_value = padding_value

        self.num_workers = min(num_workers, os.cpu_count())
        self.normalize_obj = NormalizeHub(**normalize_config)
        self.resizer = partial(normalize_shape_resize_with_padding, **resize_config)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.patch_size = getattr(self, "patch_size", 16)
        self.merge_size = getattr(self, "merge_size", 2)

    def __repr__(self):
        return (f"{self.__class__.__name__}\ndo_resize: {self.do_resize}\ndo_permute: {self.do_permute}\n"
                f"do_normalize: {self.do_normalize}\nresize_config: {self.resize_config}\n"
                f"normalize_config: {self.normalize_config}\n"
                f"num_workers: {self.num_workers}\npatch_size: {self.patch_size}\nmerge_size: {self.merge_size}\n")

    def save_pretrained(self, save_rtpath):
        config = dict(
            do_resize=self.do_resize,
            do_permute=self.do_permute,
            do_normalize=self.do_normalize,
            resize_config=self.resize_config,
            normalize_config=self.normalize_config,
            padding_value=self.padding_value,
            num_workers=self.num_workers
        )
        with open(os.path.join(save_rtpath, "img_processor.json"), 'w', encoding="utf-8") as f:
            f.write(json.dumps(config, indent=4))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "ImageProcessPacking":
        if os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            config_file = os.path.join(pretrained_model_name_or_path, "img_processor.json")
        assert os.path.exists(config_file), f"{config_file} does not exist!"
        with open(config_file, 'r', encoding="utf-8", errors="ignore") as f:
            config = json.load(f, strict=config_file)
        return cls(**config)

    def get_img_wh(self, img_pre: np.ndarray) -> tuple[int, int]:
        assert img_pre.ndim == 3, f"img_pre must be 3D, but got {img_pre.ndim}"
        if not self.do_permute:
            h, w = img_pre.shape[:2]
        else:
            h, w = img_pre.shape[-2:]
        return h, w

    def _single_preprocessing(
            self,
            _img: np.ndarray | Image.Image,
            image_shape: tuple[int, int] | None = None,
            order_id: int | None = None,
            **kwargs
    ) -> np.ndarray | tuple:
        """
        Runner order:
            step1: check whether transform
            step2: check whether resize
            step3: check whether normalize
            step4: check whether permute
        Args:
            _img:
            image_shape: tuple[int, int] | None = None,
            order_id: int | None
            **kwargs:

        Returns:

        """
        info = None

        if kwargs.get("do_resize", getattr(self, "do_resize", False)):
            assert hasattr(self, "resizer"), f"{self.__class__} does not have `resizer`"
            if image_shape is not None:
                _img = getattr(self, "resizer")(_img, image_shape=image_shape)
            else:
                _img = getattr(self, "resizer")(_img)
            if isinstance(_img, tuple):
                _img, *info = _img
                info = info if len(info) > 1 else info[0]

        if kwargs.get("do_normalize", getattr(self, "do_normalize", False)):
            assert hasattr(self, "normalize_obj"), f"{self.__class__} does not have `normalize_obj`"
            _img = getattr(self, "normalize_obj")(img=_img.astype(np.float32))
        else:
            _img = _img.astype(np.float32)
        if kwargs.get("do_permute", getattr(self, "do_permute", False)):
            _img = _img.transpose(2, 0, 1)

        if order_id is None:
            return _img if info is None else (_img, info)
        else:
            return (_img, order_id) if info is None else (_img, info, order_id)

    def preprocessing(
            self,
            img: np.ndarray | list[np.ndarray],
            image_shape: tuple[int, int] | None = None,
            to_continuous: bool = True,
            **kwargs
    ) -> list[np.ndarray]:
        """
        resize & normalize & permute
        Args:
            img (Union[np.ndarray, List[np.ndarray]]):\
            image_shape: tuple[int, int] | None = None,
            to_continuous
        Return:
            img_pre (list[np.ndarray]), (3, H, W)
        """
        if isinstance(img, np.ndarray):
            img = [img]
        img_ls = []
        img_shape_ls = []

        if len(img) == 1 or self.num_workers <= 1:
            for idx, cur_img in enumerate(img):
                info = None
                # make sure resizer is `normalize_shape_resize_with_padding`
                cur_img = self._single_preprocessing(cur_img, image_shape=image_shape, **kwargs)
                if isinstance(cur_img, tuple):
                    cur_img, *info = cur_img
                    info = info if len(info) > 1 else info[0]
                if self.patch_size is not None:
                    if self.merge_size is None:
                        self.merge_size = 2
                    factor = self.patch_size * self.merge_size
                    cur_h, cur_w = self.get_img_wh(cur_img)
                    assert cur_h % factor == 0, f"cur_h must be divisible by factor, but got {cur_h} % {factor} != 0"
                    assert cur_w % factor == 0, f"cur_w must be divisible by factor, but got {cur_w} % {factor} != 0"
                img_ls.append(cur_img)
                img_shape_ls.append(cur_img.shape)

        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(
                        self._single_preprocessing, image_shape=image_shape, _img=cur_img, order_id=order_id
                    ) for order_id, cur_img in enumerate(img)
                ]
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
                results = [i[0] if len(i) == 2 else i[:-1] for i in sorted(results, key=lambda x: x[-1])]
            info = None
            for result in results:
                if isinstance(result, tuple):
                    cur_img, *info = result
                    info = info if len(info) > 1 else info[0]
                else:
                    cur_img = result
                if self.patch_size is not None:
                    if self.merge_size is None:
                        self.merge_size = 1
                    factor = self.patch_size * self.merge_size
                    cur_h, cur_w = self.get_img_wh(cur_img)
                    assert cur_h % factor == 0, f"cur_h must be divisible by factor, but got {cur_h} % {factor} != 0"
                    assert cur_w % factor == 0, f"cur_w must be divisible by factor, but got {cur_w} % {factor} != 0"
                img_ls.append(cur_img)
                img_shape_ls.append(cur_img.shape)
        return img_ls
    
    def __call__(
            self,
            img: list[np.ndarray] | np.ndarray | list[Image.Image] | Image.Image | str | list[str],
            image_shape: tuple[int, int] | None = None,
            to_continuous: bool = True,
            **kwargs
    ) -> dict:
        """
        Return:
            dict(pixel_values=new_pixel_values, grid_hw=grid_hw)
            pixel_values: shape (total_seq_lens, c, ph, pw)
            grid_hw: shape (B, 2)
        """
        if isinstance(img, list):
            img = [to_numpy(i) for i in img]
        else:
            img = to_numpy(img)
        img_ls = self.preprocessing(img, image_shape=image_shape, to_continuous=to_continuous, **kwargs)
        grid_hw_ls = []
        new_pixel_values = []
        for cur_img in img_ls:
            C, H, W = cur_img.shape[-3:]

            grid_h = H // self.patch_size
            grid_w = W // self.patch_size
            grid_hw_ls.append((grid_h, grid_w))

            # new_pixel_value = rearrange(
            #     cur_img, 
            #     "c (gh mh ph) (hw mw pw) -> (gh hw mh mw) c ph pw", 
            #     mh=self.merge_size, 
            #     mw=self.merge_size, 
            #     gh=grid_h // self.merge_size, 
            #     hw=grid_w // self.merge_size, 
            #     ph=self.patch_size, 
            #     pw=self.patch_size
            # )
            x = cur_img.reshape(
                C,
                grid_h // self.merge_size, self.merge_size, self.patch_size,
                grid_w // self.merge_size, self.merge_size, self.patch_size
            )

            x = x.transpose(1, 4, 2, 5, 0, 3, 6)
            x = x.reshape(-1, C, self.patch_size, self.patch_size)
            new_pixel_values.append(x)
        new_pixel_values = np.concatenate(new_pixel_values, axis=0)
        grid_hw = np.array(grid_hw_ls).astype(np.int32)
        return dict(pixel_values=new_pixel_values, grid_hw=grid_hw)


# ---------------------------------------------------------------------------
# Multimodal Processing (tokenizer / placeholder / dummy data)
# ---------------------------------------------------------------------------


class _MOSSv1d5ImageProcessor:
    """Minimal image_processor attribute for call_hf_processor_mm_only compat."""

    def __init__(self, adapter: "MOSSv1d5HFProcessorAdapter") -> None:
        self._adapter = adapter

    def __call__(self, images: object = None, **kwargs: object) -> dict:
        if images is None:
            return {}
        return self._adapter._to_pixel_values(images)


class MOSSv1d5HFProcessorAdapter:
    """HF Processor-like callable for MOSS image packing + text tokenization.

    Qwen2.5-VL uses ``Qwen2_5_VLProcessor`` and never overrides
    ``BaseMultiModalProcessor._call_hf_processor``; vLLM always routes through
    ``InputProcessingContext.call_hf_processor``, which retries on fast
    tokenizer ``Already borrowed`` errors. MOSS has no official HF processor, so
    this adapter provides the same ``__call__(text=..., images=..., ...)``
    surface and returns ``BatchFeature`` like ``ProcessorMixin``.
    """

    def __init__(self, info: "MOSSv1d5ProcessingInfo") -> None:
        self._info = info
        self._img_processor: ImageProcessPacking | None = None
        self.image_processor = _MOSSv1d5ImageProcessor(self)

    def _merge_kwargs(self, *args, **kwargs) -> dict:
        """Compatibility with vLLM 0.19+ call_hf_processor_mm_only."""
        return {
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
        }

    def _get_img_processor(self) -> ImageProcessPacking:
        if self._img_processor is None:
            model_path = self._info.ctx.model_config.model
            self._img_processor = ImageProcessPacking.from_pretrained(model_path)
        return self._img_processor

    def _to_pixel_values(self, images: object) -> dict[str, object]:
        if isinstance(images, torch.Tensor):
            return (
                {"pixel_values": images.unsqueeze(0)}
                if images.ndim == 3
                else {"pixel_values": images}
            )
        if not isinstance(images, list):
            images = [images]
        processor = self._get_img_processor()
        return processor(images)

    def __call__(
        self,
        text: str,
        images: object | None = None,
        image: object | None = None,
        *,
        return_tensors: str = "pt",
        **kwargs: object,
    ) -> BatchFeature:
        del return_tensors, kwargs
        tokenizer = self._info.get_tokenizer()
        token_ids = tokenizer.encode(text)
        imgs = images if images is not None else image
        if imgs is None:
            return BatchFeature(dict(input_ids=[token_ids]), tensor_type="pt")
        res = self._to_pixel_values(imgs)
        return BatchFeature(dict(input_ids=[token_ids], **res), tensor_type="pt")


class MOSSv1d5ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(MOSSv1d5Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}  # if value is none, vLLM will set `get_limit_per_prompt() -> 999` -> supports_multimodal_inputs = True

    def get_image_size_with_most_features(self) -> tuple[int, int]:
        cfg = self.get_hf_config().encoder_config
        image_size = cfg.image_size  # get dummy image size from hf config
        if isinstance(image_size, int):
            w, h = image_size, image_size
        else:
            w, h = int(image_size[1]), int(image_size[0])
        new_h, new_w = self._align_size(h, w)
        return new_w, new_h

    def _align_size(self, h: int, w: int) -> tuple[int, int]:
        """Compute post-resize dimensions matching ImageProcessPacking logic."""

        model_path = self.ctx.model_config.model
        
        if getattr(self, "_ip_cfg", None) is None:
            ip_path = os.path.join(model_path, "img_processor.json")
            with open(ip_path) as f:
                self._ip_cfg = json.load(f)

        rc = self._ip_cfg.get("resize_config", {})
        new_h, new_w = normalize_shape(
            h, w,
            min_height_size=rc.get("min_height_size", 32),
            min_width_size=rc.get("min_width_size", 32),
            max_height_size=rc.get("max_height_size", 1024),
            max_width_size=rc.get("max_width_size", 1024),
            stride=rc.get("fixed_factor", 32),
        )
        return new_h, new_w

    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        cfg = self.get_hf_config()
        enc_cfg = cfg.encoder_config
        patch = int(enc_cfg.patch_size)
        ratio = int(cfg.compressor_config.get("ratio", 1))

        new_h, new_w = self._align_size(image_height, image_width)
        h_tokens = (new_h // patch) // ratio
        w_tokens = (new_w // patch) // ratio
        return max(h_tokens * w_tokens, 1)

    def get_hf_processor(self, **kwargs: object) -> MOSSv1d5HFProcessorAdapter:
        del kwargs
        if getattr(self, "_moss_hf_processor", None) is None:
            self._moss_hf_processor = MOSSv1d5HFProcessorAdapter(self)
        return self._moss_hf_processor


class MOSSv1d5DummyInputsBuilder(BaseDummyInputsBuilder[MOSSv1d5ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<image>" * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        del seq_len
        num_images = mm_counts.get("image", 0)
        tw, th = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=tw,
                height=th,
                num_images=num_images,
                overrides=mm_options.get("image"),
            )
        }


class MOSSv1d5MultiModalProcessor(BaseMultiModalProcessor[MOSSv1d5ProcessingInfo]):
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """
        Qwen2VL / Qwen2_5_VL do not override this: ``Qwen2VLProcessor`` already
        writes placeholder token ids into ``input_ids`` that match
        ``_get_prompt_updates``, so the base implementation can return ``True``
        for pixel inputs.

        MOSS uses ``MOSSv1d5HFProcessorAdapter`` (encode text + tensors only).
        It does not mirror HF ``input_ids`` expansion. Returning ``True`` would
        skip ``_apply_prompt_updates`` when ``mm_processor_cache_gb == 0``
        (no processor cache), breaking placeholder validation.

        A stricter parity with Qwen would be to teach the adapter to emit the
        same ``input_ids`` layout as a real HF processor, then drop this
        override for the pixel path.
        """
        if any(
            isinstance(items, (EmbeddingItems, DictEmbeddingItems))
            for items in mm_items.values()
        ):
            return super()._hf_processor_applies_updates(
                prompt_text,
                mm_items,
                hf_processor_mm_kwargs,
                tokenization_kwargs,
            )
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # del hf_inputs, hf_processor_mm_kwargs
        
        grid_hw = hf_inputs.get("grid_hw", torch.empty((0, 2)))
        image_pixel_grid_sizes = grid_hw.prod(-1)
        return dict(
            # pixel_values=MultiModalFieldConfig.batched("image"),  # what's meaning?
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", image_pixel_grid_sizes),
            grid_hw=MultiModalFieldConfig.batched("image"),
            # image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del hf_processor_mm_kwargs, out_mm_kwargs
        img_token_id = self.info.get_hf_config().decoder_config.img_token_id
        if img_token_id is None:
            raise ValueError("`decoder_config.img_token_id` is required.")

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )
            if isinstance(images, ImageEmbeddingItems):
                n = images.get_feature_size(item_idx)
            else:
                sz = images.get_image_size(item_idx)
                n = self.info.get_num_image_tokens(
                    image_width=sz.width, image_height=sz.height
                )
            return PromptUpdateDetails.select_token_id(
                [img_token_id] * n, img_token_id
            )

        return [
            PromptReplacement(
                modality="image",
                target="<image>",
                replacement=get_replacement,
            )
        ]


# ---------------------------------------------------------------------------
# Encoder: VisionTransformer
# ---------------------------------------------------------------------------


class MOSSv1d5Rotary2DEmbedding(nn.Module):
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

        out = torch.empty_like(x)
        out[..., 0::2] = out1
        out[..., 1::2] = out2
        return out

    q_rotated = apply_rotary_pos_emb(q, cos_expanded, sin_expanded)
    k_rotated = apply_rotary_pos_emb(k, cos_expanded, sin_expanded)

    return q_rotated, k_rotated


def position_meshgrid_packing_for_flash_attn(grid_hw: torch.Tensor, merge_size: int = 1) -> torch.Tensor:
    """
    Args:
        grid_hw: shape (B, 2)
    Returns:
        positions: shape (B, H * W, 2)
    """
    positions = torch.cat(
        [
            position_meshgrid_by_shape_merged(p, merge_size=merge_size)  # origin: position_meshgrid_by_shape
            for p in grid_hw
        ]
    )
    return positions


def position_meshgrid_by_shape_merged(grid_hw: torch.Tensor, merge_size: int = 1) -> torch.Tensor:
    """
    Args:
        grid_hw: shape (2, ), representing total [H/p, W/p]
        merge_size: the m value in m x m merge window
    """
    device = grid_hw.device
    H_p, W_p = grid_hw[0], grid_hw[1]
    if merge_size == 1:
        return torch.stack(
            torch.meshgrid(torch.arange(H_p, device=device), torch.arange(W_p, device=device), indexing="ij"),
            dim=-1,
        ).reshape(-1, 2).to(device)

    h1, w1 = H_p // merge_size, W_p // merge_size
    h1_grid, w1_grid, m1_grid, m2_grid = torch.meshgrid(
        torch.arange(h1, device=device), torch.arange(w1, device=device), 
        torch.arange(merge_size, device=device), torch.arange(merge_size, device=device), 
        indexing='ij'
    )
    h_coords = h1_grid * merge_size + m1_grid
    w_coords = w1_grid * merge_size + m2_grid
    
    return torch.stack([h_coords, w_coords], dim=-1).reshape(-1, 2)  #.to(device)


class MOSSv1d5VisMLP(nn.Module):
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
        self.act_fn = SiluAndMul()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class MOSSv1d5VisAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads: int = config.num_attention_heads
        self.head_dim: int = config.head_dim
        self.num_key_value_heads: int = config.num_key_value_heads
        self.repeats = self.num_attention_heads // self.num_key_value_heads
        self.scale = self.head_dim ** -0.5
        self.do_qk_norm = config.qk_norm

        if config.qk_norm:
            self.q_norm = RMSNorm(config.head_dim, eps=config.eps)
            self.k_norm = RMSNorm(config.head_dim, eps=config.eps)

        self.qkv = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim + config.num_key_value_heads * config.head_dim + config.num_key_value_heads * config.head_dim,
            bias=False
        )
        self.q_size = config.num_attention_heads * config.head_dim
        self.kv_size = config.num_key_value_heads * config.head_dim
        self.wo = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        self.attn = MMEncoderAttention(
            num_heads=config.num_attention_heads,
            head_size=config.head_dim,
            num_kv_heads=config.num_key_value_heads,
            scale=config.head_dim ** -0.5
        )

    def forward(
            self,
            x: torch.Tensor,
            pos_embed: torch.Tensor, 
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,
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
        v = v.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        
        if self.do_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # apply 2D rope
        q, k = apply_2D_rotary_emb(q, k, pos_embed)
        q = q.transpose(1, 2).contiguous() 
        k = k.transpose(1, 2).contiguous()
        v = v.contiguous()
        x = self.attn(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).flatten(-2)
        return self.wo(x)


class MOSSv1d5VisTransformerBlock(nn.Module):
    def __init__(self, config: MOSSv1d5VisionConfig):
        super().__init__()
        self.config = config
        self.attention = MOSSv1d5VisAttention(config=config)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.eps)
        self.feed_forward = MOSSv1d5VisMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # pre-norm https://arxiv.org/pdf/2002.04745
        _x = self.attention_norm(x)
        x = x + self.attention(_x, pos_embed=pos_embed, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        _x = self.ffn_norm(x)
        x = x + self.feed_forward(_x)
        return x


class MOSSv1d5VisVisionTransformerBlocks(nn.Module):
    def __init__(self, config: MOSSv1d5VisionConfig):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(MOSSv1d5VisTransformerBlock(config=config))

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return x


class MOSSv1d5VisionTransformer(nn.Module):
    def __init__(
            self,
            config: MOSSv1d5VisionConfig
    ):
        super().__init__()
        self.config = config
        self.merge_size = getattr(config, "merge_size", 2)  # make sure the merge size is 2!!!!!
        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.stride,
            bias=False,
        )
        self.rope_pos_embedding = MOSSv1d5Rotary2DEmbedding(
            dim=config.head_dim,
            height=config.grid_size[0],
            width=config.grid_size[1],
            theta=config.rope_theta
        )
        self.ln_pre = None
        self.ln_post = None
        if self.config.use_pre_norm:
            self.ln_pre = RMSNorm(config.hidden_size, eps=config.eps)

        if self.config.use_post_norm:
            self.ln_post = RMSNorm(config.hidden_size, eps=config.eps)
        self.transformer = MOSSv1d5VisVisionTransformerBlocks(config)

        if self.config.model_path is not None:
            raise NotImplementedError

    @classmethod
    def _from_config(cls, config: MOSSv1d5VisionConfig) -> "MOSSv1d5VisionTransformer":
        if isinstance(config, dict):
            return cls.from_dict(config)
        return cls(config)

    @classmethod
    def from_dict(cls, config: dict) -> "MOSSv1d5VisionTransformer":
        vit_config = MOSSv1d5VisionConfig.from_dict(config)
        return cls(vit_config)

    def forward(
            self,
            x: torch.Tensor,
            grid_hw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:  (bsz, C, H, W) or (seq_len, C, H, W) (packing mode)
        Returns:
            image_features: tensor of token features for all tokens of all images of
                shape (N_toks, D)
        """
        cu_seqlens = None
        max_seqlen = None
        if grid_hw is not None:
            seq_len = torch.cat([grid_hw.new_zeros(1), grid_hw.prod(dim=1)], dim=0)
            cu_seqlens = seq_len.cumsum(dim=0).to(torch.int32)
            max_seqlen = seq_len.max().to(torch.int32)
        
        x = self.patch_conv(x).flatten(-3).unsqueeze(0).contiguous()  # bsz, total_seq_len, dim
        pos_mesh = position_meshgrid_packing_for_flash_attn(grid_hw=grid_hw, merge_size=self.merge_size)
        pos_embed = self.rope_pos_embedding(pos_mesh).to(device=x.device)  # torch.complex
        if self.config.use_pre_norm:
            x = self.ln_pre(x)   # (bsz, seq_len, dim)
        x = self.transformer(x, pos_embed=pos_embed, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        if self.config.use_post_norm:
            x = self.ln_post(x)
        return x  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# compressor: patch merging
# ---------------------------------------------------------------------------

class MOSSv1d5PatchMergerSwiGLU(nn.Module):
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


class MOSSv1d5PatchMerger(nn.Module):
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__()
        self.hidden_size = config["input_channels"] * (config['ratio']**2)
        self.layer_norm = RMSNorm(config["input_channels"], eps=1e-6)
        self.mlp = MOSSv1d5PatchMergerSwiGLU(
            input_channels=self.hidden_size,
            intermediate_size=config["intermediate_size"],
            output_channels=config["output_channels"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.layer_norm(x).view(-1, self.hidden_size))
        return x


# ---------------------------------------------------------------------------
# Decoder: vLLM-compatible self-attention (paged KV cache)
# ---------------------------------------------------------------------------

class MOSSv1d5Attention(nn.Module):
    """MOSS self-attention built on vLLM's Attention backend.

    Differences from Qwen2Attention:
      - qk_norm is applied to the *full* (all-heads-concatenated) q / k
        vectors, matching the original MOSS training code.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 1_000_000.0,
        max_position_embeddings: int = 8192,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        attn_bias: bool = True,
        attn_o_bias: bool = False,
        partial_rope: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads

        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = max(1, num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attn_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attn_o_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.q_size , eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.kv_size, eps=rms_norm_eps)

        rope_parameters: dict[str, Any] = {
            "rope_type": "default",
            "rope_theta": rope_theta,
        }
        if partial_rope:
            rope_parameters["partial_rotary_factor"] = 0.5

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


# ---------------------------------------------------------------------------
# Decoder: SwiGLU MLP
# ---------------------------------------------------------------------------

class MOSSv1d5MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size, 
            intermediate_size * 2, 
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, 
            hidden_size, 
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_up_proj(x)
        x = self.act_fn(gate)
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class MOSSv1d5DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads

        self.self_attn = MOSSv1d5Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rope_theta=getattr(config, "rope_theta", 1_000_000.0),
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            use_qk_norm=getattr(config, "use_qk_norm", False),
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            attn_bias=getattr(config, "attn_bias", True),
            attn_o_bias=getattr(config, "attn_o_bias", False),
            partial_rope=getattr(config, "partial_rope", False),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = MOSSv1d5MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# ---------------------------------------------------------------------------
# Decoder backbone (embed_tokens + layers + final norm)
# ---------------------------------------------------------------------------
def moss_v1d5_decoder_model_invariants(
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> None:
    if inputs_embeds is not None:
        torch._check(positions.size()[0] == inputs_embeds.size()[0])
    elif input_ids is not None:
        torch._check(positions.size()[0] == input_ids.size()[0])
    if intermediate_tensors is not None:
        torch._check(
            positions.size()[0]
            == intermediate_tensors["hidden_states"].size()[0]
        )


@support_torch_compile(shape_invariants=moss_v1d5_decoder_model_invariants)
class MOSSv1d5DecoderModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        if hasattr(config, "decoder_config"):
            config = config.decoder_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            getattr(config, "tie_word_embeddings", False)
            and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.use_abs_pe = getattr(config, "abs_pe", False)
        if self.use_abs_pe:
            self.abs_pe_emb = nn.Embedding(
                num_embeddings=getattr(config, "abs_pe_max_length", 8192),
                embedding_dim=config.hidden_size,
                padding_idx=getattr(config, "pad_token_id", 0),
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MOSSv1d5DecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(
                config.hidden_size,
                eps=getattr(config, "rms_norm_eps", 1e-6),
            )
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.get_input_embeddings(input_ids)

            if self.use_abs_pe:
                hidden_states = hidden_states + self.abs_pe_emb(positions)

            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# ---------------------------------------------------------------------------
# Decoder CausalLM head
# ---------------------------------------------------------------------------

class MOSSv1d5ForCausalLM(nn.Module, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        if hasattr(config, "decoder_config"):
            dec_cfg = config.decoder_config
        else:
            dec_cfg = config

        self.config = dec_cfg
        self.model = MOSSv1d5DecoderModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = nn.Linear(
                dec_cfg.hidden_size, dec_cfg.vocab_size, bias=False
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(dec_cfg.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return logits

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


# ---------------------------------------------------------------------------
# Multimodal VL Model
# ---------------------------------------------------------------------------

@MULTIMODAL_REGISTRY.register_processor(
    MOSSv1d5MultiModalProcessor,
    info=MOSSv1d5ProcessingInfo,
    dummy_inputs=MOSSv1d5DummyInputsBuilder,
)
class MOSSv1d5VLForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """vLLM-compatible MOSS v1.6 VLM.

    encoder  →  compressor  →  language_model (MOSSv1d5ForCausalLM with paged attention)
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "decoder.": "language_model.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: MOSSv1d5Config = vllm_config.model_config.hf_config
        self.config = config
        self.encoder_config = config.encoder_config
        self.compressor_config = config.compressor_config
        self.decoder_config = config.decoder_config

        with self._mark_tower_model(vllm_config, "image"):
            self.encoder = MOSSv1d5VisionTransformer._from_config(config.encoder_config)
            self.compressor = MOSSv1d5PatchMerger(config=config.compressor_config)
            self.encoder_dtype = self.encoder.patch_conv.weight.dtype

        with self._mark_language_model(vllm_config):
            self.language_model = MOSSv1d5ForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    # ---------- multimodal helpers ----------

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> MOSSImageInputs | None:
        # print(f"================================================")
        # print(kwargs.keys())
        # print(f"================================================")
        pixel_values = kwargs.pop("pixel_values", None)
        grid_hw = kwargs.pop("grid_hw", None)
        # print(f"pixel_values: {pixel_values.shape}, grid_hw: {grid_hw.shape} {grid_hw=}")
        # print(f"================================================")
        if pixel_values is not None:
            return MOSSImagePixelInputs(
                type="pixel_values", pixel_values=pixel_values, grid_hw=grid_hw
            )
        return None
    
    def _encode(self, pixel_values, grid_hw):
        # print(">>> ", pixel_values.shape, grid_hw.shape, grid_hw)
        if self.encoder_dtype is not None and pixel_values.dtype != self.encoder_dtype:
            pixel_values = pixel_values.to(dtype=self.encoder_dtype)
        merge_size = self.compressor_config.get("ratio", 1)
        return self.compressor(self.encoder(pixel_values, grid_hw=grid_hw)).unsqueeze(0).split((grid_hw.prod(dim=1) // merge_size ** 2).tolist(), dim=1)

    def _process_image_input(
        self, image_input: MOSSImageInputs
    ) -> list[torch.Tensor]:
        """Returns a list of (1, N_tokens, D) tensors, one per image."""
        pixel_values = image_input["pixel_values"]
        grid_hw = image_input["grid_hw"]
        return self._encode(pixel_values, grid_hw)

    # ---------- SupportsMultiModal interface ----------

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        embeds_list = self._process_image_input(image_input)
        return tuple(emb.squeeze(0) for emb in embeds_list)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="compressor",
            tower_model="encoder",
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

