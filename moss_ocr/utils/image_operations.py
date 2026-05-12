"""
@author: wwjiang
"""
import functools
import os
import sys
import base64
import requests
import traceback
import logging
from io import BytesIO
from typing import Tuple, Union, Dict, Optional
import math
import numpy as np
from PIL import Image
from functools import partial
import random
import cv2


__all__ = [
    "ResizeImg",
    "remove_image_blank_edge",
    "resize_with_padding",
    "normalize_shape",
    "resize_with_padding_random_augment",
    "normalize_shape_resize_with_padding"
]

logger = logging.getLogger(__file__)


class ImageDecode:

    def __init__(self, return_mode="bgr", timeout: int = 10, logger=logger):
        """
        Args:
            return mode:
                RGB, return image format is `RGB`
                BGR, return image format is `BGR`
        parser method:
            open_url_to_ndarray
            s3_path_to_ndarray
            local_path_to_ndarray
            bytes_to_ndarray

        """
        assert return_mode.lower() in ("bgr", "rgb")
        self.return_mode = return_mode
        self.timeout = timeout
        self.logger = logger

    def open_url_to_ndarray(self, img_url, logger=None, process_rgba=False) -> np.ndarray:
        logger = self.logger if logger is None else logger
        get_request = requests.get(img_url, timeout=self.timeout)
        get_request.raise_for_status()
        bytes_data = get_request.content
        img = self.bytes_to_ndarray(bytes_data, logger=logger, process_rgba=process_rgba)
        assert isinstance(img, np.ndarray), f"img_url={img_url} is not a numpy array"
        return img

    def base64_to_ndarray(self, img_base64: str, process_rgba: bool = False, logger=None):
        """inverse read base64 file"""
        # from my_utils.io_operation import base64_file_to_array
        logger = self.logger if logger is None else logger
        img_data = base64.b64decode(img_base64)
        img_bytes = BytesIO(img_data).getvalue()
        return self.bytes_to_ndarray(img_bytes, logger=logger, process_rgba=process_rgba)

    def s3_path_to_ndarray(
            self,
            img_url: Union[str, Dict],
            logger=None,
            endpoint_url: Optional[str] = None,
            process_rgba: bool = False
    ) -> Optional[np.ndarray]:
        """
        loading s3 image path to ndarray
        Args:
            img_url (Union[str, dict]): when str: {bucket}[SEP]{storage_path};
                when dict, key1: bucket, key2: storage_path
            logger
            endpoint_url
            process_rgba
        Return:
            ndarray
        """
        import boto3
        import botocore

        logger = self.logger if logger is None else logger
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        s3_client = session.client(
            "s3",
            endpoint_url=endpoint_url,
            config=botocore.client.Config(s3={"addressing_style": "virtual"})
        )
        if isinstance(img_url, str):
            bucket, storage_path = img_url.split("[SEP]")
            assert len(bucket) > 0 and len(storage_path) > 0
        elif isinstance(img_url, dict):
            bucket = img_url.get("bucket", None)
            storage_path = img_url.get("storage_path", None)
        else:
            raise TypeError(f"img_url only support (str, dict), but received: {img_url}")
        logger.info(f"Decode image from s3 path:  BUCKET: {bucket} "
                    f"STORAGE_PATH: {storage_path}, process_rgba: {process_rgba}, endpoint_url: {endpoint_url}")
        try:
            f = BytesIO()
            s3_client.download_fileobj(bucket, storage_path, f)
            f.seek(0)
            image_byte_data = f.getvalue()
        except botocore.exceptions.ClientError as error:
            try:
                cur_session = boto3.Session(region_name=os.getenv('AWS_REGION'))
                cur_s3_client = cur_session.client("s3")
                f = BytesIO()
                cur_s3_client.download_fileobj(bucket, storage_path, f)
                image_byte_data = f.getvalue()
                logger.info(f"download image from current s3!")
            except Exception as e:
                logger.error(f"Decode image error found {e}")
                logger.error(f"{traceback.format_exc()}")
                return None
        img = self.bytes_to_ndarray(image_byte_data, logger=logger, process_rgba=process_rgba)
        return img

    def local_path_to_ndarray(self, img_url, logger=None):
        logger = self.logger if logger is None else logger
        img_bgr = cv2.imread(img_url)
        return img_bgr if self.return_mode.lower() == "bgr" else img_bgr[..., ::-1]

    def bytes_to_ndarray(self, bytes_data, logger=None, process_rgba=False):
        logger = self.logger if logger is None else logger
        if process_rgba:
            self.logger.info(f"process rgba!")
            img_bgr = self.imageIOpil_rgba(bytes_data, logger=logger)
            return img_bgr if self.return_mode.lower() == "bgr" else img_bgr[..., ::-1]
        self.logger.info(f"do not process rgba!")
        if self.isgif(bytes_data):
            img_bgr = self.imageIOpil(bytes_data, logger=logger)
        else:
            img_bgr = self.imageIOcv2(bytes_data, logger=logger)
        return img_bgr if self.return_mode.lower() == "bgr" else img_bgr[..., ::-1]

    def identity(self, img_bgr: np.ndarray, logger=None) -> np.ndarray:
        return img_bgr

    @staticmethod
    def isgif(h) -> bool:
        """GIF ('87 and '89 variants)"""
        if h[:6] in (b"GIF87a", b"GIF89a"):
            return True
        else:
            return False

    @staticmethod
    def imageIOcv2(bytes_data, logger) -> Optional[np.ndarray]:
        """return BGR image"""
        try:
            img_bgr = cv2.imdecode(
                np.asarray(bytearray(bytes_data), dtype=np.uint8), cv2.IMREAD_COLOR
            )
        except Exception as e:
            logger.warning("Error in decoding byte data into image: {}".format(e))
            return None

        if img_bgr is None:
            logger.warning("Error in decoding byte data into image")
            return None
        else:
            return img_bgr

    @staticmethod
    def imageIOpil(bytes_data, logger):
        """
        parse image from bytes to Image with bgr channel
        """
        stream = BytesIO(bytes_data)
        try:
            image_pil = Image.open(stream).convert("RGB")
        except Exception as e:
            logger.warning("PIL parse byte data to RGB image error: {}".format(e))
            try:
                image_pil_l = Image.open(stream).convert("L")
                image_pil = image_pil_l.convert("RGB")
            except Exception as e:
                logger.warning("PIL parse byte data to gray image error: {}".format(e))
                return None
        finally:
            stream.close()
        image_rgb = np.array(image_pil)
        assert image_rgb.ndim == 3
        return image_rgb[..., ::-1]  # rgb 2 bgr

    @staticmethod
    def imageIOpil_rgba(bytes_data, logger):
        stream = BytesIO(bytes_data)
        try:
            image_pil = Image.open(stream)
        except Exception as e:
            logger.error("open image bytes error: {}".format(e))
            return None
        if image_pil.mode == "RGBA":
            img = np.array(image_pil)
            img_rgb = img[..., :3]
            alpha = np.stack([img[..., -1]] * 3, axis=2) / 255
            mask = np.ones_like(img_rgb, dtype=np.uint8) * 255
            img_rgb_merge = np.clip(img_rgb * alpha + mask * (1 - alpha), 0, 255)
            return img_rgb_merge[..., ::-1].astype(np.uint8)
        else:
            try:
                image_pil = image_pil.convert("RGB")
            except Exception as e:
                logger.warning("PIL parse byte data to RGB image error: {}".format(e))
                try:
                    image_pil_l = Image.open(stream).convert("L")
                    image_pil = image_pil_l.convert("RGB")
                except Exception as e:
                    logger.warning("PIL parse byte data to gray image error: {}".format(e))
                    return None
            finally:
                stream.close()
            image_rgb = np.array(image_pil)
            assert image_rgb.ndim == 3
            return image_rgb[..., ::-1]  # rgb 2 bgr


class ResizeImg:
    def __init__(self, **kwargs):
        """
        Example:
        >>>img = (np.random.rand(512, 768, 3) * 255).astype(np.uint8)
        >>>resize_params = [
            dict(params=dict(fixed_height=1024, fixed_factor=16), output_shape=(1024, 1536, 3)),
            dict(params=dict(fixed_width=1024, fixed_factor=16), output_shape=(688, 1024, 3)),
            dict(params=dict(limit_long=1024, fixed_factor=16), output_shape=(512, 768, 3)),
            dict(params=dict(fixed_long=1024, fixed_factor=16), output_shape=(688, 1024, 3)),
            dict(params=dict(limit_long=1024, fixed_factor=16, min_size=600), output_shape=(608, 768, 3)),
            dict(params=dict(limit_long_height=600, fixed_factor=16, max_size=700), output_shape=(608, 704, 3)),
            dict(params=dict(limit_long_width=600, fixed_factor=16, max_size=700), output_shape=(400, 608, 3)),
            dict(params=dict(limit_short=600, fixed_factor=16), output_shape=(512, 768, 3)),
            dict(params=dict(fixed_short=600, fixed_factor=16), output_shape=(608, 912, 3)),
            dict(params=dict(limit_short=600, fixed_factor=16, max_size=700), output_shape=(608, 704, 3)),
            dict(params=dict(limit_short_height=600, fixed_factor=16, max_size=700), output_shape=(608, 704, 3)),
            dict(params=dict(limit_short_width=600, fixed_factor=16, max_size=700), output_shape=(512, 768, 3)),
            dict(params=dict(fixed_shape=(320, 632), fixed_factor=16), output_shape=(320, 640, 3)),
            dict(params=dict(fixed_shape=(320, 632), fixed_factor=16, min_size=350), output_shape=(352, 640, 3)),
        ]

        kwargs:
            ########### 11 种resize模式 ###########
            fixed_height: 固定高度
            fixed_width: 固定宽度
            fixed_shape: 固定尺寸resize
            fixed_short: 最短边固定
            fixed_long: 最长边固定
            limit_short: 最短边不低于
            limit_short_height: 图片高度不低于
            limit_short_width: 图片宽度不低于
            limit_long: 最长边不超过
            limit_long_height: 图片高度不超过
            limit_long_width: 图片宽度不超过

            ###########可选参数###########
            fixed_factor: resize尺寸需要包含的因数（确保尺寸能被该数整除，向上取整）
            min_size: 图片最小尺寸不低于
            max_size： 图片最长尺寸不高于
            ###########默认参数###########
        """

        super(ResizeImg, self).__init__()
        self.min_size = kwargs.get("min_size", None)
        self.max_size = kwargs.get("max_size", None)
        self.fixed_factor = kwargs.get("fixed_factor", 1)
        self.interpolation = kwargs.get("interpolation", cv2.INTER_LINEAR)
        self.fixed_len = 0
        self.reshape_size = None
        self.kwargs = kwargs

        if "padding_value" in kwargs:
            # TODO: testing
            assert "image_shape" in kwargs
            image_shape = kwargs["image_shape"]
            if isinstance(image_shape, int):
                image_shape = (image_shape, image_shape)
            assert isinstance(image_shape, tuple | list)
            self.resize_shape = kwargs["image_shape"]
            kwargs["return_crop_info"] = True
            self.resize_func = functools.partial(resize_with_padding, **kwargs)

        if "fixed_shape" in kwargs:
            self.reshape_size = kwargs['fixed_shape']
            self.resize_func = self.resize_image

        elif "limit_short" in kwargs:
            self.resize_func = self.resize_image_by_limit_short
            self.fixed_len = kwargs.get("limit_short")

        elif "limit_long" in kwargs:
            self.resize_func = self.resize_image_by_limit_long
            self.fixed_len = kwargs.get("limit_long")

        elif "fixed_short" in kwargs:
            self.resize_func = self.resize_image_by_fixed_short
            self.fixed_len = kwargs.get("fixed_short")

        elif "fixed_long" in kwargs:
            self.resize_func = self.resize_image_by_fixed_long
            self.fixed_len = kwargs.get("fixed_long")

        elif "fixed_height" in kwargs:
            self.resize_func = self.resize_image_by_fixed_height
            self.fixed_len = kwargs.get("fixed_height")

        elif "fixed_width" in kwargs:
            self.resize_func = self.resize_image_by_fixed_width
            self.fixed_len = kwargs.get("fixed_width")

        elif "limit_long_height" in kwargs:
            self.resize_func = self.resize_image_by_limit_long_height
            self.fixed_len = kwargs.get("limit_long_height")

        elif "limit_long_width" in kwargs:
            self.resize_func = self.resize_image_by_limit_long_width
            self.fixed_len = kwargs.get("limit_long_width")

        elif "limit_short_height" in kwargs:
            self.resize_func = self.resize_image_by_limit_short_height
            self.fixed_len = kwargs.get("limit_short_height")

        elif "limit_short_width" in kwargs:
            self.resize_func = self.resize_image_by_limit_short_width
            self.fixed_len = kwargs.get("limit_short_width")

        else:
            self.resize_func = self.resize_image_by_adaptive

    def __call__(self, img: np.ndarray, return_crop_info: bool = False) -> np.ndarray | tuple[np.ndarray, list | tuple]:
        ori_h, ori_w = img.shape[:2]
        resize_params = dict()
        resize_params["img"] = img
        resize_params["fixed_len"] = self.fixed_len
        resize_params["reshape_size"] = self.reshape_size
        resize_params["fixed_factor"] = self.fixed_factor
        resize_params["min_size"] = self.min_size
        resize_params["max_size"] = self.max_size
        resize_img, resize_crop_info = self.resize_func(**resize_params)
        if return_crop_info:
            return resize_img, resize_crop_info
        else:
            return resize_img

    def resize_image_by_fixed_width(self, img: np.ndarray, fixed_len: int, **kwargs):
        """图片宽固定为fixed_len。若fix_factor不为1，则将图片宽固定为 math.ceil(fixed_len / fixed_factor) * fixed_len
        """
        h, w = img.shape[:2]
        resize_w = fixed_len
        resize_h = fixed_len * float(h) / w
        reshape_size = (resize_h, resize_w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)

    def resize_image_by_fixed_height(self, img: np.ndarray, fixed_len: int, **kwargs):
        """图片高固定为fixed_len。若fix_factor不为1，则将图片高固定为 math.ceil(fixed_len / fixed_factor) * fixed_len
        """
        h, w = img.shape[:2]
        resize_h = fixed_len
        resize_w = fixed_len * float(w) / h
        reshape_size = (resize_h, resize_w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)

    def resize_image_by_fixed_short(self, img: np.ndarray, fixed_len: int, **kwargs):
        """最短边固定为fixed_len。若fix_factor不为1，则将最短边固定为 math.ceil(fixed_len / fixed_factor) * fixed_len
        """
        h, w = img.shape[:2]
        if h < w:
            # h 为短边
            resize_h = fixed_len
            resize_w = fixed_len * float(w) / h
        else:
            resize_w = fixed_len
            resize_h = fixed_len * float(h) / w
        reshape_size = (resize_h, resize_w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)

    def resize_image_by_fixed_long(self, img: np.ndarray, fixed_len: int, **kwargs):
        """最长边固定为fixed_len。若fix_factor不为1，则将最短边固定为 math.ceil(fixed_len / fixed_factor) * fixed_len
        """
        h, w = img.shape[:2]
        if h > w:
            # h 为长边
            resize_h = fixed_len
            resize_w = fixed_len * float(w) / h
        else:
            resize_w = fixed_len
            resize_h = fixed_len * float(h) / w
        reshape_size = (resize_h, resize_w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)

    def resize_image_by_limit_short(self, img: np.ndarray, fixed_len: int, **kwargs):
        """最短边不小于fixed_len。若fix_factor不为1，则将最短边固定为 math.ceil(fixed_len / fixed_factor) * fixed_len
        """
        h, w = img.shape[:2]
        if h < w:
            # h 为短边
            resize_h = h if h > fixed_len else fixed_len
            resize_w = resize_h * float(w) / h
        else:
            resize_w = w if w > fixed_len else fixed_len
            resize_h = resize_w * float(h) / w
        reshape_size = (resize_h, resize_w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)

    def resize_image_by_limit_short_height(self, img: np.ndarray, fixed_len: int, **kwargs):
        """图片高度最小不低于fixed_len。若fix_factor不为1，则将最短边固定为 math.ceil(fixed_len / fixed_factor) * fixed_len
        """
        h, w = img.shape[:2]
        resize_h = h if h > fixed_len else fixed_len
        resize_w = resize_h * float(w) / h
        reshape_size = (resize_h, resize_w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)

    def resize_image_by_limit_short_width(self, img: np.ndarray, fixed_len: int, **kwargs):
        """图片宽度最小不低于fixed_len。若fix_factor不为1，则将最短边固定为 math.ceil(fixed_len / fixed_factor) * fixed_len
        """
        h, w = img.shape[:2]
        resize_w = w if w > fixed_len else fixed_len
        resize_h = resize_w * float(h) / w
        reshape_size = (resize_h, resize_w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)

    def resize_image_by_limit_long(self, img: np.ndarray, fixed_len: int, **kwargs):
        """最长边不超过fixed_len。若fix_factor不为1，则将最短边固定为 round(fixed_len / fixed_factor) * fixed_len
        """
        h, w = img.shape[:2]
        if h > w:
            # h 为长边
            resize_h = fixed_len if h > fixed_len else h
            resize_w = resize_h * float(w) / h
        else:
            resize_w = fixed_len if w > fixed_len else w
            resize_h = resize_w * float(h) / w
        reshape_size = (resize_h, resize_w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)

    def resize_image_by_limit_long_height(self, img: np.ndarray, fixed_len: int, **kwargs):
        """图片高度不超过fixed_len。若fix_factor不为1，则将最短边固定为 round(fixed_len / fixed_factor) * fixed_len
        """
        h, w = img.shape[:2]
        resize_h = fixed_len if h > fixed_len else h
        resize_w = resize_h * float(w) / h
        reshape_size = (resize_h, resize_w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)

    def resize_image_by_limit_long_width(self, img: np.ndarray, fixed_len: int, **kwargs):
        """图片宽度不超过fixed_len。若fix_factor不为1，则将最短边固定为 round(fixed_len / fixed_factor) * fixed_len
        """
        h, w = img.shape[:2]
        resize_w = fixed_len if w > fixed_len else w
        resize_h = resize_w * float(h) / w
        reshape_size = (resize_h, resize_w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)

    def resize_image_by_adaptive(self, img: np.ndarray, fixed_len: int, **kwargs):
        """根据当前尺寸基于固定因数resize
        """
        h, w = img.shape[:2]
        reshape_size = (h, w)
        del kwargs["reshape_size"]
        return self.resize_image(img, reshape_size=reshape_size, **kwargs)

    @staticmethod
    def resize_image(
            img: np.ndarray,
            reshape_size: Tuple,
            fixed_factor: int = 1,
            **kwargs
    ) -> Tuple:
        img_dtype = img.dtype
        ori_h, ori_w = img.shape[:2]
        resize_h, resize_w, *_ = reshape_size
        resize_h = int(math.ceil(max(resize_h / fixed_factor, 1)) * fixed_factor)
        resize_w = int(math.ceil(max(resize_w / fixed_factor, 1)) * fixed_factor)

        min_size = kwargs["min_size"]
        max_size = kwargs["max_size"]
        if isinstance(min_size, (int, float)) and min_size > 0:
            min_size = int(math.ceil(max(min_size / fixed_factor, 1)) * fixed_factor)
            resize_h = max(min_size, resize_h)
            resize_w = max(min_size, resize_w)
        if isinstance(max_size, (int, float)) and max_size > 0:
            max_size = int(math.ceil(max(max_size / fixed_factor, 1)) * fixed_factor)
            resize_h = min(max_size, resize_h)
            resize_w = min(max_size, resize_w)
        assert isinstance(resize_h, int) and isinstance(resize_w, int)
        # print(resize_w, resize_h)
        resize_img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        return resize_img.astype(img_dtype), [0, resize_img.shape[0], 0, resize_img.shape[1], ratio_h, ratio_w]


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


def resize_with_padding_random_augment(
        img: np.ndarray,
        image_shape: tuple[int | None, int | None] | int,
        padding_value: int | list = 255,
        keep_aspect_ratio: bool | list[bool] = True,
        center_pad: bool | list[bool] = True,
        no_scale_up: bool | list[bool] = True,
        return_crop_info: bool = False,
        hw_shift_range: float | list[float] | None = None,
        hw_shift_p: float = 0,
        interpolation: int = cv2.INTER_LINEAR,

):
    if hw_shift_range is not None and hw_shift_p >= random.random():
        if isinstance(hw_shift_range, float):
            assert hw_shift_range > 0
            minima_shift = -hw_shift_range
            maxima_shift = hw_shift_range
        else:
            minima_shift, maxima_shift = hw_shift_range[:2]
        h, w = img.shape[:2]
        h_ratio = random.uniform(minima_shift, maxima_shift) + 1
        w_ratio = random.uniform(minima_shift, maxima_shift) + 1
        new_h, new_w = int(h * h_ratio), int(w * w_ratio)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if isinstance(padding_value, list):
        padding_value = random.sample(padding_value, 1)[0]

    if isinstance(keep_aspect_ratio, list):
        keep_aspect_ratio = random.sample(keep_aspect_ratio, 1)[0]

    if isinstance(center_pad, list):
        center_pad = random.sample(center_pad, 1)[0]

    if isinstance(no_scale_up, list):
        no_scale_up = random.sample(no_scale_up, 1)[0]

    return resize_with_padding(
        img,
        image_shape=image_shape,
        padding_value=padding_value,
        keep_aspect_ratio=keep_aspect_ratio,
        center_pad=center_pad,
        no_scale_up=no_scale_up,
        return_crop_info=return_crop_info,
        interpolation=interpolation,
        backend=random.sample(["opencv", "PIL"], 1)[0]
    )


def remove_image_blank_edge(
        img: np.ndarray,
        return_coord: list = False
) -> np.ndarray | tuple[np.ndarray, list[int]]:
    """
    Args:
        img:
        return_coord: bool: whether return clip coordinate
    Returns:
        if return_coord:
            img_remove_blank (np.ndarray): image without edge
        else:
            img_remove_blank (np.ndarray): image without edge
            clip_loc (List[int]): [xmin, ymin, xmax, ymax]

    """

    def get_blank_idx(pixel_sum: list, thresh: int = 1) -> Tuple[int, int]:
        index_info = [idx for idx, i in enumerate(pixel_sum) if i >= thresh]
        if index_info.__len__() < 2:
            start = 0
            end = len(pixel_sum)
        else:
            start = index_info[0]
            end = index_info[-1]
        return start, end

    assert isinstance(img, np.ndarray)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_canny = cv2.Canny(img_gray, 50, 150)  # 0为背景, 255为edge
    # return img_canny
    img_canny[img_canny[:] < 128] = 0
    img_canny[img_canny[:] >= 128] = 1

    row_sum = list(np.sum(img_canny, axis=0))  # 行和
    col_sum = list(np.sum(img_canny, axis=1))  # 列和

    row_start, row_end = get_blank_idx(col_sum)
    col_start, col_end = get_blank_idx(row_sum)

    img_remove_blank = img[row_start: row_end, col_start: col_end, ...]
    if not return_coord:
        return img_remove_blank
    else:
        return img_remove_blank, [col_start, row_start, col_end, row_end]


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
        min_height_size / height if height < min_height_size else 0,
        min_width_size / width if width < min_width_size else 0
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
        **kwargs
) -> np.ndarray | tuple[np.ndarray, tuple]:
    ori_h, ori_w = img.shape[:2]
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
    )


def img_grid_draw(
        img_ls: list[np.ndarray, str, Image.Image],
        n_cols: int = 10,
        maximum: int = 100,
        grid_size: tuple[int, int] = (224, 224)
) -> np.ndarray:

    rows = []
    idx = 0
    img_num = min(len(img_ls), maximum)
    delta = (n_cols - img_num % n_cols) % n_cols
    for i in range(math.ceil(img_num / n_cols)):
        cols = []
        for j in range(n_cols):
            cols.append(resize_with_padding(img_ls[idx], (224, 224)))
            idx += 1
            if idx >= img_num:
                cols.extend([np.zeros((*grid_size, 3), dtype=np.uint8) for _ in range(delta)])
                break
        rows.append(np.concatenate(cols, axis=1))
    final_img = np.concatenate(rows, axis=0)
    final_img[::grid_size[0]-1, ...] = (114, 114, 114)
    final_img[:, ::grid_size[0]-1, ...] = (114, 114, 114)
    return final_img


class NormalizeShape:
    def __inti__(self, candidate_shape_arr):
        self.candidate_shape_arr = candidate_shape_arr

    def allocate_cluster_center(
            self,
            height: int,
            width: int,
            stride: int = 16,
            min_height_size: int = 32,
            min_width_size: int = 32,
            max_height_size: int = 1344,
            max_width_size: int = 1344,
    ) -> tuple[int, int]:
        """
        Args:
            height: int
            width: int
            stride: int = 16,
            min_height_size: int = 32,
            min_width_size: int = 32,
            max_height_size: int = 1344,
            max_width_size: int = 1344,
        Returns:
        """

        # step1: find nearest sequence group
        height, width = normalize_shape(
            height=height,
            width=width,
            min_height_size=min_height_size,
            min_width_size=min_width_size,
            max_height_size=max_height_size,
            max_width_size=max_width_size,
            stride=stride
        )
        cur_hw = np.array([[height, width]])  # (1, 2)

        dist = np.sum(abs(cur_hw - self.candidate_shape_arr), axis=-1)  # (N, )
        alloc_h, alloc_w = self.candidate_shape_arr[np.argmin(dist)].tolist()
        alloc_h = math.ceil(alloc_h / stride) * stride
        alloc_w = math.ceil(alloc_w / stride) * stride
        return (alloc_h, alloc_w)


if __name__ == "__main__":
    import random
    from my_utils.io_operation import get_all_file_path, checkdir
    from PIL import Image

    curdir = os.path.dirname(__file__)
    rtpath = os.path.join(curdir, "../..")
    img_rtpath = os.path.join(rtpath, f"static/testing_images")
    img_path_ls = get_all_file_path(img_rtpath)

    for img_path in img_path_ls[:2]:
        img = cv2.imread(img_path)
        res, crop_info = resize_with_padding(
            img,
            image_shape=(1024, 2048),
            center_pad=False,
            no_scale_up=True,
            padding_value=114,
            return_crop_info=True
        )
        start_h, end_h, start_w, end_w, *_ = crop_info
        ori_img = res[start_h: end_h, start_w: end_w]
        Image.fromarray(res).show()
        print(f"ori h, w: {img.shape}, output shape: {res.shape} {ori_img.shape}")
