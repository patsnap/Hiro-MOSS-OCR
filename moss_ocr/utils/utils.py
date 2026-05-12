"""
@author: wwjiang
"""

import os
import re
import sys
import traceback
from typing import Union
import subprocess
import time
import math
import csv
from io import BytesIO
import hashlib
import yaml
import json
import copy
import base64
from PIL import Image
import numpy as np
import cv2
import socket
import logging
from tqdm import tqdm
from logging.handlers import RotatingFileHandler
import inspect
import types
import torch
import torch.nn as nn
import importlib
from collections import OrderedDict
from inspect import isfunction, isclass
from functools import partial
import random
from prettytable import PrettyTable
from typing import Union, Optional, Callable, Any, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import chain

curdir = os.path.dirname(__file__)
rtpath = os.path.join(curdir, "../..")
sys.path.append(rtpath)

_logger = logging.getLogger(__name__)


def has_chinese(text: str) -> bool:
    # 中文汉字范围：4E00–9FFF
    return re.search(r'[\u4e00-\u9fff]', text) is not None


def has_kana(text: str) -> bool:
    pattern = r'[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\uFF66-\uFF9D]'
    return re.search(pattern, text) is not None



def get_console_logger():
    console_logger = logging.getLogger("console_logger")
    if not console_logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_logger.addHandler(console_handler)
    
    console_logger.setLevel(logging.INFO)
    return console_logger


def extract_markdown_math(md_text: str) -> tuple[list, list]:
    if not md_text:
        return [], []
    block_pattern = re.compile(r"\$\$(.*?)\$\$|\\\[(.*?)\\\]", re.DOTALL)
    inline_pattern = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)|\\\((.+?)\\\)")
    blocks_raw = block_pattern.findall(md_text)
    blocks = [b[0] if b[0] else b[1] for b in blocks_raw]
    inlines_raw = inline_pattern.findall(md_text)
    inlines = [i[0] if i[0] else i[1] for i in inlines_raw]
    return blocks, inlines


def to_numpy(img: np.ndarray | Image.Image | str) -> np.ndarray:
    if isinstance(img, str):
        assert os.path.exists(img)
        return cv2.imread(img)[..., ::-1]
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, Image.Image):
        return np.array(img.convert('RGB'))[..., :3]
    raise TypeError(f"Unsupported type {type(img)}")


def to_pil(img: np.ndarray | Image.Image | str) -> Image.Image:
    if isinstance(img, str):
        assert os.path.exists(img)
        return Image.open(img).convert('RGB')
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    if isinstance(img, Image.Image):
        return img.convert('RGB')
    raise TypeError(f"Unsupported type {type(img)}")


def ddp_flag() -> bool:
    """
    Return:
        True: ddp
        False: no ddp
    """
    return int(os.environ.get('RANK', -1)) != -1


def check_installation(package):
    try:
        __import__(package)
        return True
    except ImportError:
        return False


def read_base64_file(file_path: str) -> str:
    """read file base64 string
    Args:
        file_path (str): file path
    Returns:
        str: base64 string
    """
    with open(file_path, "rb") as f:
        file_base64 = base64.b64encode(f.read())
    return file_base64.decode('utf-8')



def array_to_bytes(x: np.ndarray) -> bytes:
    """convert ndarray to byte
    Args:
        x (np.ndarray) : ndarray
    Returns:
        bytes
    """
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()


def bytes_to_array(b: bytes) -> np.ndarray:
    """inverse array_to_bytes
    Args:
        b (bytes): bytes
    Returns:
        np.ndarray
    """
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def str2hashint(text, digit = 16) -> int:
    """convert str 2 hash scala"""
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % 10 ** digit


def init_callable_from_config(
        config: dict | None,
        logger=_logger
) -> Union[Callable, None, Any]:

    if config is None:
        return None

    def smart_init(_obj, _params: dict) -> Optional[Callable]:
        if _obj is None:
            return None
        if isclass(_obj):
            return _obj(**_params)
        elif isfunction(_obj):
            return partial(_obj, **_params)
        elif callable(_obj):
            try:
                return _obj(**_params)
            except TypeError:
                return partial(_obj, **_params)
        else:
            logger.error(f"Unsupported object type: {type(_obj)}")
            return None

    obj_name = config["name"]
    init_params = config.get("params", {})
    obj = None
    if not isinstance(init_params, dict):
        logger.error(f"`params` must be a dict, got {type(init_params)}")
        return None

    module_name: str | None = config.get("module_name", None)
    if module_name is None:
        obj = globals().get(obj_name, None)
        if obj is None:
            logger.error(f"No object named `{obj_name}` found in global scope")
            return None
    else:
        try:
            obj = getattr(importlib.import_module(module_name), obj_name)
        except ModuleNotFoundError as e:
            logger.error(f"Module `{module_name}` not found: {e}")
            return None
        except AttributeError as e:
            logger.error(f"Object `{obj_name}` not found in module `{module_name}`: {e}")
            return None
    return smart_init(_obj=obj, _params=init_params)


dynamic_init_from_config = init_callable_from_config


def replace_key_attr_recurrence(_d: dict, key: str, target_value, inplace=False) -> dict:
    if not inplace:
        _d = copy.deepcopy(_d)
    for k, v in _d.items():
        if k == key:
            _d[k] = target_value
            continue
        if isinstance(v, dict):
            replace_key_attr_recurrence(v, key, target_value, inplace=True)
    return _d


# def set_sig_params_by_config(func: types.FunctionType, config: object):
#     sig = inspect.signature(func)
#     sig_params = sig.parameters
#     output_dict = dict()
#     for i in sig_params:
#         if i in self.__dict__ and i not in self.__ignore_attribution__:
#             output_dict[i] = self.__dict__[i]
#     return output_dict
#
#     pass


def cal_time(t_start: float) -> float:
    """calculate time consume. From t_start to current time.
    """
    return time.perf_counter() - t_start


def cal_time_ms(t_start: float) -> float:
    """calculate time consume. From t_start to current time. millisecond"""
    return cal_time(t_start) * 1000


def get_dir(_path: str) -> str:
    """
    Get directory, Example:
    >>> get_dir("/path/to/dir/file_name.suffix")
    >>> "/path/to/dir"
    Args:
        _path (str): file_path
    Returns:
        (str): file dir path
    """
    return os.path.split(_path)[0]


def get_prefix(_path: str) -> str:
    """
    Get file_name prefix, Example:
    >>> get_dir("/path/to/dir/file_name.suffix")
    >>> "file_name"
    Args:
        _path (str): file_path
    Returns:
        (str): file prefix name
    """
    return '.'.join(os.path.basename(_path).split('.')[:-1])


def get_suffix(_path: str) -> str:
    """
    Get file_name suffix, Example:
    >>> get_dir("/path/to/dir/file_name.suffix")
    >>> "suffix"
    Args:
        _path (str): file_path
    Returns:
        (str): file suffix name
    """
    return os.path.basename(_path).split('.')[-1]


def get_basename(_path: str) -> str:
    """
    Get file_name, Example:
    >>> get_dir("/path/to/dir/file_name.suffix")
    >>> "file_name.suffix"
    Args:
        _path (str): file_path
    Returns:
        (str): file name
    """
    return os.path.basename(_path)


def load_json(json_path: str, strict: bool = True) -> any:
    """convert json file to dict
    Args:
        json_path: json path
        strict: bool, whether strict mode
    Returns:
        json info
    """
    with open(json_path, 'r', encoding="utf-8", errors="ignore") as f:
        json_info = json.load(f, strict=strict)
    return json_info


def load_jsonl(jsonl_path: str) -> list[dict]:
    """convert jsonl file to list of dict
    Args:
        jsonl_path: jsonl path
    Returns:
        list of dict
    """
    with open(jsonl_path, 'r', encoding="utf-8", errors="ignore") as f:
        return [json.loads(line.strip()) for line in f]


def to_json(input_dict: any, json_path: str, indent: int = 4) -> None:
    """convert info to json file
    Args:
        input_dict: dict
        json_path: json path
        indent (int)
    Returns:
        None
    """
    json_str = json.dumps(input_dict, indent=indent)
    with open(json_path, 'w', encoding="utf-8") as f:
        f.write(json_str)


def load_yaml_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    global_config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return global_config


def save_dict_to_yaml(info: dict, save_path: str) -> None:
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(info, f, allow_unicode=True)


def is_gpu_available():
    try:
        # 在 Windows 上使用 'where'，在 Unix/Linux 上使用 'which'
        subprocess.run(['which', 'nvidia-smi'], check=True, stdout=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        try:
            subprocess.run(['which', 'nvcc'], check=True, stdout=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            return False


def str_to_bool(s: Union[str, bool]) -> bool:
    if str(s).lower() == "true":
        return True
    else:
        return False


class Timer:
    def __init__(self):
        self.start = time.time()
        self.records = {}
        self.total = 0

    def elapsed(self):
        end = time.time()
        res = end - self.start
        self.start = end
        return res

    def record(self, category, extra_time=0):
        e = self.elapsed()
        if category not in self.records:
            self.records[category] = 0

        self.records[category] += e + extra_time
        self.total += e + extra_time

    def summary(self, minima_elapsed: float = 0.0001, return_type: str = "string") -> str:

        table = PrettyTable(align="l")
        table.field_names = ["step", "time_consume(s)", "ratio(%)"]

        for k, v in self.records.items():
            if v >= minima_elapsed:
                table.add_row([k, f"{v:.4f}", f"{v * 100 / self.total:.2f}"])
        table.add_row(["Total", f"{self.total:.4f}", "-"])

        match return_type:
            case "string":
                res = f"{self.total:.4f}s"
                additions = [x for x in self.records.items() if x[1] >= minima_elapsed]
                if not additions:
                    return res

                res += " ("
                res += ", ".join([f"{category}: {time_taken:.4f}s" for category, time_taken in additions])
                res += ")"
                return res

            case "table":
                return table.get_string()

            case "html":
                return table.get_html_string()


def dict2str_hierarchy(d: dict, logger=None, sep='\n') -> str:
    """
    print dict info with format
    Args:
        d(Dict): dict
        sep: connect char
        logger
    Returns:
        dict_info_str(str)
    Examples:
    >>> demo_d = dict(a=1, b=dict(c=1, d='apple'))
    >>> dict2str_hierarchy(demo_d)
    -a
      |--(int)1
    -b
      -c
        |--(int)1
      -d
        |--(str)apple"
    >>> dict2str_hierarchy(d, sep=' ')
    -a  |--(int)1 -b    -c...
    """
    dict_info = []
    depth = 0
    tap_str: str = "  "

    def _dict2str_bk(_d, _depth):
        key_indent = ''.join([tap_str for _ in range(_depth)]) if _depth > 0 else ''
        for k, v in _d.items():
            candidate = [(v, _depth)]
            dict_info.append(f"{key_indent}-{k}")
            while candidate:
                cur_item, _depth = candidate.pop()
                if not isinstance(cur_item, dict):
                    cur_indent = ''.join([tap_str for _ in range(_depth + 1)])
                    cur_item_type = repr(type(cur_item)).split("'")[-2]
                    dict_info.append(f"{cur_indent}|--({cur_item_type}){cur_item}")
                else:
                    _dict2str_bk(cur_item, _depth+1)
    _dict2str_bk(d, depth)
    if logger is not None:
        for i in dict_info:
            logger.info(i)
    out_str = sep.join(dict_info)
    del dict_info
    return out_str


def get_all_file_path(
        rtpath,
        file_suffix_ls: list | tuple = ("jpg", "jpeg", "png", "gif", "jfif", "PNG", "JPEG", "JPG", "tif"),
        max_recurrent_deep=10000
) -> list:
    """
    递归rtpath目录及其子目录，保存目录下所有后缀在file_siffix_ls的文件
    Args:
        rtpath: str, 递归的目标目录
        file_suffix_ls: List, 需要保存的文件类型,
            when "*" in file_suffix_ls or not isinstance(file_suffix_ls, list) return all file path
        max_recurrent_deep: 最大递归深度
    Returns: List, 所有目标文件的绝对路径
    """
    total_file_path = []
    for rtdir, dirname, files in os.walk(rtpath):
        cur_subdir = rtdir.split(rtpath)[-1].strip(os.sep).strip()
        if cur_subdir == "":
            cur_level = 0
        else:
            cur_level = cur_subdir.split(os.sep).__len__()
        if cur_level > max_recurrent_deep:
            continue
        if "*" in file_suffix_ls or file_suffix_ls == "*":
            total_file_path.extend([os.path.join(rtdir, i) for i in files])
        else:
            total_file_path.extend([os.path.join(rtdir, i) for i in files if get_suffix(i).lower() in file_suffix_ls])
    return total_file_path


def get_all_dir_path(rtpath: str, max_recurrent_deep=1000):
    """
        递归rtpath目录及其子目录，保存目录下递归深度小于max_recurrent_deep的所有路径
        Args:
            rtpath: str, 递归的目标目录
            file_suffix_ls: List, 需要保存的文件类型
            max_recurrent_deep: 最大递归深度

        Returns: List, 所有目标文件的绝对路径
        """

    recurrent_deep_info = {}
    tgt_path = [rtpath]
    total_dir_path = [rtpath]
    recurrent_deep_info[rtpath] = 1
    while tgt_path:
        cur_path = tgt_path.pop(0)
        cur_dir_deep = recurrent_deep_info[cur_path]
        if cur_dir_deep > max_recurrent_deep:
            continue
        cur_item_ls = os.listdir(cur_path)
        for cur_item in cur_item_ls:
            cur_item_path = os.path.join(cur_path, cur_item)
            if os.path.isfile(cur_item_path):
                continue
            elif os.path.isdir(cur_item_path):
                total_dir_path.append(cur_item_path)
                recurrent_deep_info[cur_item_path] = recurrent_deep_info[cur_path] + 1
                tgt_path.append(cur_item_path)
                continue
            else:
                continue
    return total_dir_path


def checkdir(path: list | str) -> None:
    """check whether path exist, if not, create it!
    Args:
        path (Union[str, List[str]]): path or path list
    Returns:
        None
    """

    def single_dir_check(_path):
        if not os.path.exists(_path):
            os.makedirs(_path, exist_ok=True)
        else:
            pass

    if isinstance(path, str):
        single_dir_check(path)
        return
    elif isinstance(path, (list, tuple)):
        _ = [single_dir_check(i) for i in path]
    else:
        raise TypeError


def decode_text_by_line(
        txt_path: str,
        strip_item: list | tuple | None = (" ", "\t", "\n", "\r\n"),
        ignore_item_ls: list | tuple = (),
        replace_item_dict: dict | None = None
        ) -> list[str]:
    """read text file then return by line"""
    line_info = []
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.readlines()
        for line in content:
            line_clean = process_text(
                line, strip_item=strip_item, ignore_item_ls=ignore_item_ls, replace_item_dict=replace_item_dict
                )
            if len(line_clean) > 0:
                line_info.append(line_clean)
            else:
                continue
    return line_info


def process_text(
        text: str,
        strip_item: list | tuple | None = None,
        ignore_item_ls: list | tuple = (),
        replace_item_dict: dict | None = None
        ) -> str:
    """process text with custom rule"""
    replace_item_dict = replace_item_dict if replace_item_dict is not None else {}
    strip_item = strip_item if strip_item is not None else []
    for i in strip_item:
        text = text.strip(i)
    for char_i in ignore_item_ls:
        text = text.replace(char_i, '')
    for k, v in replace_item_dict.items():
        text = text.replace(k, v)
    return text


def chunk_list_to_list(chunk_ls: list[list]) -> tuple[list[list], list[int]]:
    out_ls = []
    chunk_size_ls = []
    for i in chunk_ls:
        out_ls.extend(i)
        chunk_size_ls.append(len(i))
    return out_ls, chunk_size_ls


def list_to_chunk_list(ls: list, chunk_size_ls: list[int]) -> list[list]:
    _chunk_list = []
    cum_id = 0
    for i in chunk_size_ls:
        _chunk_list.append(ls[cum_id: cum_id + i])
        cum_id += i
    return _chunk_list


def chunk_list(
        ls: list,
        chunk_size: int,
        padding: bool = False,
        resample: bool = False,
        fill_value=None
) -> list[list]:
    """
    Args:
        ls:
        chunk_size:
        padding: if `ls % chunk_size != 0`, padding ls
        resample: if True, when `ls % chunk_size != 0`, fill ls by sample ls
        fill_value: when `ls % chunk_size != 0` and `resample = False` fill ls by fill_value

    Returns:
        List[List]
    """
    length = len(ls)
    if padding:
        delta = chunk_size - (length % chunk_size)
        if resample:
            fill_value = random.choices(ls, k=delta)
        else:
            fill_value = [fill_value] * delta
        ls += fill_value
    return [ls[idx: idx + chunk_size] for idx in range(0, len(ls), chunk_size)]


def get_file_complete_path(file_name: str, file_rtpath: str) -> str | None:
    if file_rtpath is not None:

        prior_path = os.path.join(file_rtpath, file_name)
        if os.path.exists(prior_path):
            return prior_path

        file_suffix = get_suffix(file_name)
        file_name = get_basename(file_name)
        file_ls = get_all_file_path(file_rtpath, file_suffix_ls=[file_suffix])

        if len(file_ls) <= 0:
            return None

        for i in file_ls:
            if get_basename(i) == file_name:
                return os.path.abspath(i)

        if os.path.exists(file_name):
            return os.path.abspath(file_name)
    else:
        if os.path.exists(file_name):
            return os.path.abspath(file_name)
    return None


class PlainLogger:
    """
    mock logger
    """
    def __init__(self):
        pass

    def info(self, s):
        pass

    def debug(self, s):
        pass

    def warn(self, s):
        pass

    def error(self, s):
        pass


def get_logger(log_file, backupcount=3, maximum_log_file_size=50, stream=False) -> logging.Logger:
    """

    Args:
        log_file: log file path
        backupcount: backup number
        maximum_log_file_size: maximum file size
        stream: whether equip with stream handler

    Return:
        logging.Logger: 配置好的日志记录器
    """
    # 使用文件名作为logger名称，确保唯一性
    logger_name = os.path.basename(log_file)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 移除已有的处理器，避免重复
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s-%(filename)s[line:%(lineno)d]-%(process)d-%(thread)d-%(levelname)s-%(message)s"
    )

    # 创建文件处理器
    maxbytes = maximum_log_file_size * 1024 * 1024
    # 不删除已存在的日志文件，而是追加或轮换
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=maxbytes,
        backupCount=backupcount,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if stream:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


class LoggerShowInfo:
    def __init__(self):
        self._container = OrderedDict()

    @property
    def container(self) -> OrderedDict:
        return self._container.copy()

    def unique_key_value_insert(self, key: str, value: Any):
        if not isinstance(key, str):
            raise TypeError(f"Key must be str, got {type(key)}")
        self._container[key] = value

    def unique_dict_insert(self, info: Dict[str, Any]):
        for key, value in info.items():
            self._container[key] = value

    def __len__(self):
        return len(self._container)

    def __contains__(self, key: str) -> bool:
        return key in self._container

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self._container.get(key, default)

    def __repr__(self) -> str:
        str_ls = []
        for key, value in self._container.items():
            if isinstance(value, float):
                str_ls.append(f"{key}: {value:.4f}")
            elif isinstance(value, torch.Tensor):
                str_ls.append(f"{key}: {value.item():.4f}")
            elif value is None:
                str_ls.append(f"{key}: None")
            else:
                str_ls.append(f"{key}: {value}")
        return " ".join(str_ls)

    def reset(self):
        self._container = OrderedDict()


class SmoothQueue:
    def __init__(self, name, smooth_window=10):

        self.smooth_window = smooth_window
        self.queue = []
        self.q_len = 0
        self.name = name
        self.global_mean = 0
        self.cur_item = 0
        self.global_count = 0

    @property
    def value(self) -> float:
        return sum(self.queue) / max(self.q_len, 1e-8)

    def update(self, item: Union[int, float, torch.Tensor]) -> None:
        if isinstance(item, torch.Tensor):
            item = item.item()
        self.global_count += 1
        self.cur_item = item
        self.update_global_mean()
        if self.q_len >= self.smooth_window:
            self.queue.pop(0)
            self.queue.append(item)
        else:
            self.queue.append(item)
            self.q_len += 1

    def update_global_mean(self):
        self.global_mean = (self.global_mean * (self.global_count - 1) + self.cur_item) / max(self.global_count, 1e-8)

    def clear(self):
        self.queue = []
        self.q_len = 0

    def __repr__(self):
        return f"{self.name}: {self.value:.4f}"

    def __len__(self):
        return self.q_len


class CheckpointManager:

    def __init__(
            self,
            keep_n_last: int = 1,
            keep_n_best: int = 1,
            higher_better: bool = True,
            logger=_logger
    ):
        self.keep_n_last = max(0, keep_n_last)
        self.keep_n_best = max(0, keep_n_best)
        self.higher_better = higher_better
        self.flag = 1 if self.higher_better else -1

        self.last_of_n_queue: list[tuple] = []  # [(global_iteration, checkpoint_path), ]
        self.best_of_n_queue: list[tuple] = []  # [(metric_value, checkpoint_path), ]

        self.logger = logger

    @property
    def best_of_n_ckpt(self) -> set:
        return {i[1] for i in self.best_of_n_queue}

    @property
    def last_of_n_ckpt(self) -> set:
        return {i[1] for i in self.last_of_n_queue}

    def _safe_remove(self, _path):
        if os.path.exists(_path):
            try:
                os.remove(_path)
            except Exception as e:
                self.logger.error(f"remove file {_path} error found: {e}")

    def save_checkpoint_info(self, save_rtpath: str) -> None:
        checkdir(save_rtpath)
        save_path = os.path.join(save_rtpath, f"checkpoint_manager_info.json")
        save_info = dict(
            higher_better=self.higher_better,
            last_of_n_queue=self.last_of_n_queue,
            best_of_n_queue=self.best_of_n_queue
        )
        to_json(save_info, save_path)

    def update(self, ckpt_path: str, iterations: int, metric: float) -> None:
        # update last of N
        if self.keep_n_last > 0:
            if len(self.last_of_n_queue) < self.keep_n_last:
                self.last_of_n_queue.append((iterations, ckpt_path))
            else:
                _, rm_ckpt_path = self.last_of_n_queue.pop(0)
                if rm_ckpt_path not in self.best_of_n_ckpt and os.path.exists(rm_ckpt_path):
                    self._safe_remove(rm_ckpt_path)
                self.last_of_n_queue.append((iterations, ckpt_path))

        if self.keep_n_best > 0:  # update best of N
            weighted_metric = metric * self.flag
            if len(self.best_of_n_queue) < self.keep_n_best:
                self.best_of_n_queue.append((weighted_metric, ckpt_path))
                self.best_of_n_queue.sort(key=lambda x: x[0])
            else:
                if weighted_metric > self.best_of_n_queue[0][0]:
                    _, rm_ckpt_path = self.best_of_n_queue.pop(0)
                    if rm_ckpt_path not in self.last_of_n_ckpt and os.path.exists(rm_ckpt_path):
                        self._safe_remove(rm_ckpt_path)
                    self.best_of_n_queue.append((weighted_metric, ckpt_path))
                    self.best_of_n_queue.sort(key=lambda x: x[0])

    def __repr__(self):
        return f'CheckpointManager: \nlast_of_n_queue={self.last_of_n_queue}\nbest_of_n_queue={self.best_of_n_queue}'


class TrainingLogWriter:
    """TODO: maybe harmful training efficiency"""
    def __init__(self, output_dir: str):
        self.write = csv.writer(output_dir)
        self.row_numer = 0

    def log(self, logger_info: LoggerShowInfo):
        if self.row_numer == 0:
            self.write.writerow(logger_info.container.keys())
        self.write.writerow(logger_info.container.values())


def truncate_repetitions(text: str, min_len=15):
    # From nougat, with some cleanup
    if len(text) < 2 * min_len:
        return text

    # try to find a length at which the tail is repeating
    max_rep_len = None
    for rep_len in range(min_len, int(len(text) / 2)):
        # check if there is a repetition at the end
        same = True
        for i in range(0, rep_len):
            if text[len(text) - rep_len - i - 1] != text[len(text) - i - 1]:
                same = False
                break

        if same:
            max_rep_len = rep_len

    if max_rep_len is None:
        return text

    lcs = text[-max_rep_len:]

    # remove all but the last repetition
    text_to_truncate = text
    while text_to_truncate.endswith(lcs):
        text_to_truncate = text_to_truncate[:-max_rep_len]

    return text[:len(text_to_truncate)]


def ndarray2base64str_cv2(ndarray_vector: np.ndarray) -> str:
    """invert func is `base64_file_to_array`
    Args:
        ndarray_vector (np.ndarray): np.ndarray
    Returns:
        str: base64 string
    """
    img_str = cv2.imencode('.png', ndarray_vector)[1].tostring()
    img_base64 = base64.b64encode(img_str).decode('utf-8')
    return img_base64


def get_host_ip() -> str:
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        st.connect(('10.255.255.255', 1))
        host_ip = st.getsockname()[0]
    except Exception as e:
        print(e)
        host_ip = '127.0.0.1'
    finally:
        st.close()
    return host_ip


def otsl_checking(table_str):
    if "<fcel>" in table_str or "<nl>" in table_str:
        return True
    else:
        return False


def multi_process_runner(
        func: Callable,
        params_ls: list[Any],
        num_workers: int = 8,
        backend: str = 'thread',
        show_progress: bool = True,
        unpack_params: bool = False
) -> list[Any]:
    """
    Parallelize function execution while preserving input order.

    Args:
        func: Target function to execute.
        params_ls: List of parameters for the function.
        num_workers: Max workers (default: 8).
        backend: 'thread' or 'process' (default: 'thread').
        show_progress: Show progress bar (default: True).
        unpack_params: If True, unpack parameters as:
        - list/tuple: func(*param)
        - dict: func(**param)
        - else: func(param)

    Returns:
        Results in the same order as params_ls (failed tasks return None).
    """
    assert backend in ('thread', 'process'), "backend must be 'thread' or 'process'"
    num_workers = min(num_workers, os.cpu_count() or 1)
    num_workers = max(num_workers, 1)

    runner = ThreadPoolExecutor if backend == 'thread' else ProcessPoolExecutor
    with runner(max_workers=num_workers) as executor:
        if not unpack_params:
            futures = {executor.submit(func, param): i for i, param in enumerate(params_ls)}
        else:
            futures = {}
            for i, param in enumerate(params_ls):
                if isinstance(param, (list, tuple)):
                    futures[executor.submit(func, *param)] = i
                elif isinstance(param, dict):
                    futures[executor.submit(func, **param)] = i
                else:
                    futures[executor.submit(func, param)] = i

        results = [None] * len(params_ls)  # 预分配结果列表
        iter_futures = tqdm(as_completed(futures), total=len(futures)) if show_progress else as_completed(futures)

        for future in iter_futures:
            idx = futures[future]  # 获取原始索引
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Task {idx} failed: {str(e)} {traceback.format_exc()}")
                results[idx] = None

    return results


def to_numpy(img: np.ndarray | Image.Image | str) -> np.ndarray:
    if isinstance(img, str):
        assert os.path.exists(img)
        _img = cv2.imread(img)
        if _img is None:
            return np.array(Image.open(img).convert('RGB'))[..., :3]
        else:
            img = _img
        return img[..., ::-1]
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, Image.Image):
        return np.array(img.convert('RGB'))[..., :3]
    raise TypeError(f"Unsupported type {type(img)}")


def flat_nested_list(nested_list: list[list]) -> list:
    """flatten nested list"""
    flat_list = list(chain.from_iterable(nested_list))
    return flat_list


def robust_latex_to_md(text: str) -> str:
    """
    LaTeX -> Markdown 转换 (v3 修复版)
    修复：v2 版本过度优化导致公式内不能包含 (2n-1) 或矩阵 [matrix] 的问题。
    """
    
    # 核心修正：
    # group(4) 和 group(6) 的否定字符集改为 [^\\\\]
    # 含义：只要不是反斜杠，都是合法内容。
    # 如果遇到反斜杠，交由 \\. 来消耗。
    # 这样既保留了 (2n-1) 的合法性，又防止了回溯。
    
    token_pattern = re.compile(r'''
        (```[\s\S]*?```|`[^`]*`)          |  # Group 1: 代码块
        (\\\\)                            |  # Group 2: 消耗双反斜杠
        (\\\[((?:\\.|[^\\\\])*?)\\\])     |  # Group 3: Display Math \[ ... \]
        (\\\(((?:\\.|[^\\\\])*?)\\\))        # Group 4: Inline Math \( ... \)
    ''', re.VERBOSE | re.DOTALL)

    def replacement(match):
        if match.group(1): return match.group(1)
        if match.group(2): return match.group(2)
        
        # Display Math
        if match.group(3):
            content = match.group(4)
            return f'\n$$\n{content.strip()}\n$$\n'

        # Inline Math
        if match.group(5):
            content = match.group(6)
            return f'${content.strip()}$'
            
        return match.group(0)

    return token_pattern.sub(replacement, text)
    

if __name__ == "__main__":
    # pdf_obj = PDFOperationHub(
    #     "/Users/jiangweiwei/Downloads/2010_Using Fast Weights to Improve Persistent Contrastive Divergence.pdf",
    #     batch_size=2
    # )
    # print(pdf_obj.__len__())
    _nested_list = [[1, 2], [2], [3, 4, 5]]
    res = chunk_list_to_list(_nested_list)
    print(res)
    out = list_to_chunk_list(*res)
    print(out)

    print(flat_nested_list(_nested_list))
