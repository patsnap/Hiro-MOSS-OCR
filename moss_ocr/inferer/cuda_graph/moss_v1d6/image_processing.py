import os
import math
import json
from dataclasses import dataclass
from typing import Dict, Optional
from functools import partial
from concurrent.futures import as_completed, ThreadPoolExecutor
from PIL import Image
import cv2
import numpy as np
import torch


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

    def __getitem__(self, key):
        return getattr(self, key)


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
        return (img - 1).astype(np.float32)

    @staticmethod
    def general_normalize_v1(img: np.ndarray, **kwargs) -> np.ndarray:
        img /= 255.0
        return img.astype(np.float32)

    @staticmethod
    def general_normalize(img: np.ndarray, **kwargs) -> np.ndarray:
        """normalize image to [0, 1] by ((img / 255.) - 0.5) / 0.5"""
        img = img / 255.0
        img -= 0.5
        img /= 0.5
        return img.astype(np.float32)

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
        img = img / 255.0
        img = (img - np.array(mean)) / np.array(std)
        return img.astype(np.float32)


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


def resize_with_padding(
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
    # print(new_h, new_w)
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


class ImageProcess:
    """
    Base class for image process
        processing order: transform -> resize -> normalize -> permute
    """
    def __init__(
            self,
            do_resize: bool = False,
            do_permute: bool = False,
            do_normalize: bool = False,
            resize_config: dict | None = None,
            normalize_config: dict | None = None,
            padding_value: int = 255,
            num_workers: int = 8
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

    def __repr__(self):
        return (f"{self.__class__.__name__}\ndo_resize: {self.do_resize}\ndo_permute: {self.do_permute}\n"
                f"do_normalize: {self.do_normalize}\nresize_config: {self.resize_config}\n"
                f"normalize_config: {self.normalize_config}\n"
                f"num_workers: {self.num_workers}\n")

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

    def __call__(
            self,
            img: list[np.ndarray] | np.ndarray | list[Image.Image] | Image.Image | str | list[str],
            image_shape: tuple[int, int] | None = None,
            to_continuous: bool = True,
            **kwargs
    ) -> ImageProcessOutput:
        """batch input -> batch output"""
        if isinstance(img, list):
            img = [to_numpy(i) for i in img]
        else:
            img = to_numpy(img)
        res = self.preprocessing(
            img=img, image_shape=image_shape, to_continuous=to_continuous, **kwargs
        )
        if isinstance(res, tuple):
            img = res[0]
        else:
            img = res
        return ImageProcessOutput(pixel_values=torch.from_numpy(img))

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
    ) -> np.ndarray | tuple[np.ndarray, list]:
        """
        resize & normalize & permute
        Args:
            img (Union[np.ndarray, List[np.ndarray]]):\
            image_shape: tuple[int, int] | None = None,
            to_continuous
            timer(Timer | None)
        Return:
            img_pre (np.ndarray), (B, 3, H, W)
        """
        if isinstance(img, np.ndarray):
            img = [img]
        img_ls = []
        img_shape_ls = []
        info_ls = []

        if len(img) == 1 or self.num_workers <= 1:
            for idx, cur_img in enumerate(img):
                info = None
                cur_img = self._single_preprocessing(cur_img, image_shape=image_shape, **kwargs)
                if isinstance(cur_img, tuple):
                    cur_img, *info = cur_img
                    info = info if len(info) > 1 else info[0]

                info_ls.append(info)
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
                info_ls.append(info)
                img_ls.append(cur_img)
                img_shape_ls.append(cur_img.shape)

        if len(set(img_shape_ls)) > 1:
            padding_value = kwargs.get("padding_value", getattr(self, "padding_value", None))
            assert padding_value is not None, f"You should setting `padding_value`"

            if kwargs.get("do_normalize", self.do_normalize):
                dtype = np.float32
            else:
                dtype = img_ls[0].dtype

            img_batch = np.ones(
                shape=np.max(np.array(img_shape_ls), axis=0).tolist(),
                dtype=dtype
            ) * padding_value

            if kwargs.get("do_normalize", getattr(self, "do_normalize", False)):
                assert hasattr(self, "normalize_obj"), f"{self.__class__} does not have `normalize_obj`"

                if kwargs.get("do_permute", getattr(self, "do_permute", False)):
                    img_batch = img_batch.transpose(1, 2, 0)
                    img_batch = getattr(self, "normalize_obj")(img_batch)
                    img_batch = img_batch.transpose(2, 0, 1)
                else:
                    img_batch = getattr(self, "normalize_obj")(img_batch)

            img_batch = np.stack([img_batch] * len(img_shape_ls), axis=0)
            for idx, cur_img in enumerate(img_ls):
                cur_shape = cur_img.shape
                img_batch[idx, :cur_shape[0], :cur_shape[1], :cur_shape[2]] = cur_img
            return img_batch if info_ls[0] is None else (img_batch, info_ls)
        else:
            if to_continuous:
                return np.ascontiguousarray(np.stack(img_ls)) if info_ls[0] is None \
                    else (np.ascontiguousarray(np.stack(img_ls)), info_ls)
            else:
                return np.stack(img_ls) if info_ls[0] is None else (np.stack(img_ls), info_ls)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "ImageProcess":
        subfolder = kwargs.get("subfolder")
        if os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            config_dir = os.path.join(pretrained_model_name_or_path, subfolder) if subfolder else pretrained_model_name_or_path
            config_file = os.path.join(config_dir, "img_processor.json")
        else:
            from transformers.utils import cached_file

            cache_kwargs = {
                key: kwargs[key]
                for key in (
                    "cache_dir",
                    "force_download",
                    "local_files_only",
                    "revision",
                    "subfolder",
                    "token",
                )
                if key in kwargs and kwargs[key] is not None
            }
            if "use_auth_token" in kwargs and "token" not in cache_kwargs:
                cache_kwargs["token"] = kwargs["use_auth_token"]
            config_file = cached_file(pretrained_model_name_or_path, "img_processor.json", **cache_kwargs)
        assert config_file is not None and os.path.exists(config_file), f"{config_file} does not exist!"
        with open(config_file, 'r', encoding="utf-8", errors="ignore") as f:
            config = json.load(f, strict=True)
        return cls(**config)


def unite_test_nearest_image_shape():
    height = 334
    width = 224
    nearest_n = 3
    alloc_h_w_ls = get_nearest_image_shape(height, width, nearest_n)
    print(alloc_h_w_ls)


if __name__ == "__main__":
    unite_test_nearest_image_shape()
