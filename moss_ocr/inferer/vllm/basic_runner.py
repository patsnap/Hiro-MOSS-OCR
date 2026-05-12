import os
import logging
import re
from tkinter.constants import E
import cv2
import weakref
import threading
import time 
import math
from dataclasses import dataclass
from typing import Union, List, Optional, Sequence, Any
import requests
from PIL import Image
import numpy as np
import asyncio
import random
import traceback
import functools
import httpx
from concurrent.futures import ProcessPoolExecutor
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI, OpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    APIStatusError,
    BadRequestError,
    InternalServerError,
    AuthenticationError
)
from openai.types.chat.chat_completion import ChatCompletion
from abc import ABC, abstractmethod
from moss_ocr.utils.utils import (
    multi_process_runner, 
    chunk_list, 
    flat_nested_list, 
    truncate_repetitions, 
    truncate_repetitions_fast_slice,
    to_numpy
)
from moss_ocr.utils.image_operations import get_nearest_image_shape, get_img_hw


_logger = logging.getLogger(__name__)

@dataclass 
class OCRResult:
    result: Any
    task: str | None 
    is_succeed: bool
    time_cost: float | None
    model_name: str | None = None
    is_repeated: bool | None = None
    error_message: str | None = None
    ppl: float | None = None
    finish_reason: str | None = None
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class BaseVllmPipeline(ABC):
    TASK_PROMPT_MAP = ...
    DEFAULT_MODEL_PATH = ...
    BACKEND = "cuda_graph"

    @staticmethod
    def to_numpy(img: np.ndarray | Image.Image | str) -> np.ndarray:
        if isinstance(img, str):
            assert os.path.exists(img)
            return cv2.imread(img)[..., ::-1]
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            return np.array(img.convert('RGB'))[..., :3]
        raise TypeError(f"Unsupported type {type(img)}")

    def __init__(self, model_path: str, max_length: int = 2048, logger=_logger, **kwargs):
        self.model_path = model_path
        self.max_length = max_length
        self.logger = logger
        ...

    @abstractmethod
    def run(self, img: np.ndarray | str | Image.Image, task: str, max_length: int | None = None, **kwargs) -> OCRResult:
        ...

    @abstractmethod
    def run_batch(
            self,
            img_ls: list[np.ndarray] | list[Image.Image],
            task_ls: list[str] | str,
            max_length_ls: int | list[int] | None = None,
            **kwargs
    ) -> list[OCRResult]:
        ...


class BaseVllmPipelineOpenAI(ABC):
    TASK_PROMPT_MAP: dict = dict()
    BACKEND = "vllm"
    URL = ...
    MODEL_NAME = ...
    API_KEY = ...
    MAX_LENGTH = ...

    def __init__(
        self,
        model_path: str | None = None,
        max_length: int | None = None,
        max_retry: int = 3,
        max_concurrent: int = 32,
        max_concurrent_for_processing: int = 32,
        url: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        detect_repeat: bool = True,
        logger=_logger,
        **kwargs
    ):
        self.url = url or self.URL
        self.api_key = api_key or self.API_KEY
        self.model_name = model_path or self.MODEL_NAME
        self.max_length = max_length or self.MAX_LENGTH

        self.timeout = timeout
        self.max_retry = max_retry
        self.max_concurrent = max_concurrent
        self.max_concurrent_for_processing = max_concurrent_for_processing
        self.process_pool = ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, max_concurrent_for_processing))

        self.logger = logger or _logger

        self._current_loop = None
        self._http_client = None
        self._async_client = None
        self.detect_repeat = detect_repeat
        self._url_index = 0
        self._async_clients = []

    def _get_client(self) -> AsyncOpenAI:
        loop = asyncio.get_running_loop()
        
        if self._current_loop is not loop:
            self._current_loop = loop
            self._url_index = 0
            
            limits = httpx.Limits(
                max_connections=self.max_concurrent + 10,
                max_keepalive_connections=self.max_concurrent
            )
            self._http_client = httpx.AsyncClient(limits=limits, timeout=self.timeout)
            if "[SEP]" not in self.url:
                self._async_clients = [AsyncOpenAI(
                    base_url=self.url,
                    api_key=self.api_key,
                    http_client=self._http_client,
                    max_retries=0 
                )]
            else:
                url_ls = self.url.split("[SEP]")
                self._async_clients = [AsyncOpenAI(
                    base_url=url,
                    api_key=self.api_key,
                    http_client=self._http_client,
                    max_retries=0 
                ) for url in url_ls
            ]
        client = self._async_clients[min(self._url_index, len(self._async_clients) - 1)]
        self._url_index = (self._url_index + 1) % len(self._async_clients)
        return client
        
    @property
    def support_task(self) -> tuple[str]:
        return tuple(self.TASK_PROMPT_MAP.keys())

    @abstractmethod
    def build_payload(
        self,
        img: Union[np.ndarray, str, Image.Image],
        task: str,
        max_length: int | None = None,
        **kwargs
    ) -> dict:
        raise NotImplementedError("Subclass must implement this method")

    def postprocessing(self, result, task, **kwargs) -> Any:
        return result

    def calculate_ppl(self, response) -> float | None:
        logprobs = response.choices[0].logprobs
        if logprobs is None:
            return None 
        else:
            if logprobs.content is None:
                return None
            logprobs_per_token = [float(i.logprob) for i in logprobs.content]
            if len(logprobs_per_token) == 0:
                return None
            ppl = math.exp(-sum(logprobs_per_token) / len(logprobs_per_token))
            return ppl
    
    async def async_run(
            self, 
            img: Union[np.ndarray, str, Image.Image], 
            task: str, 
            max_length: int | None = None, 
            **kwargs
        ) -> OCRResult:

        max_length = max_length or self.max_length
        if task not in self.support_task:
            return OCRResult(
                result=None,
                is_succeed=False,
                task=task,
                time_cost=0,
                model_name=self.model_name,
                error_message=f"Task {task} is not supported, {self.model_name} only support: {self.support_task}",
            )

        _t = time.perf_counter()
        try:
            for i in range(self.max_retry):
                client = self._get_client()
                try:
                    payload = await asyncio.to_thread(self.build_payload, img, task, max_length=max_length)
                    response: ChatCompletion = await client.chat.completions.create(**payload)
                    _result = response.choices[0].message.content
                    _finish_reason = response.choices[0].finish_reason
                    _total_tokens = getattr(response.usage, "total_tokens", None)
                    _prompt_tokens = getattr(response.usage, "prompt_tokens", None)
                    _completion_tokens = getattr(response.usage, "completion_tokens", None)

                    is_repeated = False
                    if self.detect_repeat and _result is not None and _finish_reason != "stop":
                        loop = asyncio.get_running_loop()
                        _truncate_result = await loop.run_in_executor(
                            self.process_pool, 
                            truncate_repetitions_fast_slice,
                              _result
                        ) 
                        if _truncate_result != _result:
                            is_repeated = True
                        else:
                            is_repeated = False
                        _result = _truncate_result
                    result = await asyncio.to_thread(self.postprocessing, _result, task, **kwargs)
                    
                    return OCRResult(
                        result=result, 
                        task=task, 
                        is_succeed=True, 
                        time_cost=time.perf_counter() - _t, 
                        model_name=self.model_name, 
                        is_repeated=is_repeated,
                        finish_reason=_finish_reason,
                        total_tokens=_total_tokens,
                        prompt_tokens=_prompt_tokens,
                        completion_tokens=_completion_tokens
                    )
                
                except RateLimitError as e:
                    sleep_time = min((2 ** i), 60) + random.uniform(0, 1)
                    self.logger.warning(f"[Attempt {i+1}/{self.max_retry}] Rate limit exceeded. Retrying in {sleep_time:.2f}s: {e}")
                    await asyncio.sleep(sleep_time) 
                
                except (APIConnectionError, APITimeoutError, InternalServerError) as e:
                    self.logger.warning(f"[Attempt {i+1}/{self.max_retry}] Network/Server error. Retrying... : {e}")
                    await asyncio.sleep(random.uniform(1, 3))
                
                except httpx.HTTPStatusError as e:
                    self.logger.warning(f"[Attempt {i+1}/{self.max_retry}] HTTP Status Error: {e.response.status_code} - {e.response.text}")
                    await asyncio.sleep(random.uniform(1, 3))

                except BadRequestError as e:
                    if "Already borrowed" in str(e):
                        self.logger.warning(f"Retrying due to borrowed error: {e}")
                        await asyncio.sleep(random.uniform(0.5, 1.5))
                        continue
                    else:
                        self.logger.error(f"Bad Request Error - NOT retrying: {e}")
                        raise
            
                except AuthenticationError as e:
                    self.logger.error(f"Authentication Error - NOT retrying: {e}")
                    raise
                
                except APIStatusError as e:
                    self.logger.error(f"OpenAI API Status Error: {e.status_code} - {e.response}")
                    raise
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
                    raise
                    
            raise RuntimeError(f"OpenAI inference failed after {self.max_retry} retries")
        
        except Exception as e:
            runner_time = time.perf_counter() - _t
            self.logger.error(f"OpenAI inference failed after {self.max_retry} retries: {e}\n{traceback.format_exc()}")
            return OCRResult(
                result=None, 
                task=task, 
                is_succeed=False, 
                time_cost=runner_time, 
                model_name=self.model_name, 
                error_message=str(e),
            )

    def run(self, img: Union[np.ndarray, str, Image.Image], task: str, max_length: int | None = None, **kwargs) -> OCRResult:
        return asyncio.run(self.async_run(img, task, max_length, **kwargs))

    async def async_run_batch(self, img_ls: list, task_ls: list, max_length_ls: list, **kwargs) -> list[OCRResult]:
        sem = asyncio.Semaphore(self.max_concurrent)

        async def bounded_run(img, task, length):
            async with sem:
                return await self.async_run(img, task, max_length=length, **kwargs)

        tasks = [
            bounded_run(img, task, length)
            for img, task, length in zip(img_ls, task_ls, max_length_ls)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=False) 
        return results

    def run_batch(
            self, img_ls: list, 
            task_ls: Union[list[str], str], 
            max_length_ls: Union[int, list[int], None] = None, 
            **kwargs
        ) -> list[OCRResult]:
        if "max_length" in kwargs:
            max_length_ls = kwargs.pop("max_length")
        
        assert isinstance(img_ls, list), f"img_ls must be a list, but got {type(img_ls)}"
        
        if isinstance(task_ls, str):
            task_ls = [task_ls] * len(img_ls)
        
        if len(task_ls) != len(img_ls):
            raise ValueError(f"task_ls must be the same length as img_ls, but got {len(task_ls)} and {len(img_ls)}")

        max_length_ls = max_length_ls or [self.max_length] * len(img_ls)
        if isinstance(max_length_ls, int):
            max_length_ls = [max_length_ls] * len(img_ls)
        
        if len(max_length_ls) != len(img_ls):
            raise ValueError("max_length_ls must be the same length as img_ls")

        return asyncio.run(self.async_run_batch(img_ls, task_ls, max_length_ls, **kwargs))

    def close(self):
        if hasattr(self, 'process_pool') and self.process_pool:
            self.process_pool.shutdown(wait=False)
        if self._http_client and not self._http_client.is_closed:
            pass

    def __del__(self):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    async def aclose(self):
        """异步关闭网络客户端和进程池"""
        if self._http_client and getattr(self._http_client, "is_closed", False) is False:
            await self._http_client.aclose()
        if self.process_pool:
            self.process_pool.shutdown(wait=False)


class VllmSpawnRoundRobin:
    def __init__(self, runners: list[BaseVllmPipelineOpenAI] | None = None) -> None:
        if not runners:
            raise ValueError("SpawnRoundRobin must be at least one runner")
        self._runners = tuple(runners)
        self._i = 0
        self._lock = asyncio.Lock()
    
    @property
    def max_concurrent(self) -> int:
        return self._runners[0].max_concurrent

    def __len__(self) -> int:
        return len(self._runners)

    async def async_run(self, img: Any, task: str, max_length: int | None = None, **kwargs: Any) -> OCRResult:
        async with self._lock:
            r = self._runners[self._i % len(self._runners)]
            self._i += 1
        return await r.async_run(img, task, max_length=max_length, **kwargs)
    
    async def async_run_batch(self, img_ls: list, task_ls: list, max_length_ls: list | None = None, **kwargs: Any) -> list[OCRResult]:
        async with self._lock:
            r = self._runners[self._i % len(self._runners)]
            self._i += 1
        if max_length_ls is None:
            max_length_ls = [None] * len(img_ls)
        return await r.async_run_batch(img_ls, task_ls, max_length_ls=max_length_ls, **kwargs)

    def run(self, img: Any, task: str, max_length: int | None = None, **kwargs: Any) -> OCRResult:
        return asyncio.run(self.async_run(img, task, max_length=max_length, **kwargs))

    def run_batch(self, img_ls: list, task_ls: list, max_length_ls: list | None = None, **kwargs: Any) -> list[OCRResult]:
        return asyncio.run(self.async_run_batch(img_ls, task_ls, max_length_ls=max_length_ls, **kwargs))


def build_round_robin_ocr_runner(
    urls: list[str], 
    runner_cls: BaseVllmPipelineOpenAI,
    num_instances: int,
    max_concurrent: int | None = None, 
    **kwargs: Any
):
    runners = []
    for i in range(num_instances):
        url = urls[i % len(urls)]
        _kwargs: dict = {"url": url}
        if max_concurrent is not None:
            _kwargs.update({"max_concurrent": max_concurrent})
        runners.append(runner_cls(**_kwargs, **kwargs))
    spawn_round_robin = VllmSpawnRoundRobin(runners)
    return spawn_round_robin


async def sweep_task(
    runner: VllmSpawnRoundRobin | BaseVllmPipelineOpenAI, 
    img_ls: list[str], 
    task_ls: list[str], 
    max_length_ls: list | None = None, 
    batch_size: int | None = None,
    max_concurrent: int = 32,
    rollback_on_repeat: bool = False, 
    n_nearest: int = 3,
    logger=_logger, 
    **kwargs: Any
) -> list[OCRResult]:
    batch_size = batch_size or runner.max_concurrent

    assert batch_size > 0, f"batch_size must be greater than 0, but got {batch_size}"
    assert n_nearest > 0, f"n_nearest must be greater than 0, but got {n_nearest}"
    assert max_concurrent > 0, f"max_concurrent must be greater than 0, but got {max_concurrent}"
    assert len(img_ls) > 0, f"img_ls must be a non-empty list, but got {len(img_ls)}"
    assert len(task_ls) > 0, f"task_ls must be a non-empty list, but got {len(task_ls)}"

    sem = asyncio.Semaphore(max_concurrent)

    async def run_one_batch(img_ls: list, task_ls: list, max_length_ls: list) -> list[OCRResult]:
        async with sem:
            return await runner.async_run_batch(img_ls, task_ls, max_length_ls, **kwargs)
    
    def chunk_inputs(_img_ls, _task_ls, _max_length_ls, _batch_size):
        assert len(_img_ls) == len(_task_ls) == len(_max_length_ls), f"_img_ls and _task_ls must have the same length, but got {len(_img_ls)}, {len(_task_ls)}, {len(_max_length_ls)}"
        _chunk_size = math.ceil(len(_img_ls) / _batch_size)
        _img_chunk_ls = chunk_list(_img_ls, _chunk_size)
        _task_chunk_ls = chunk_list(_task_ls, _chunk_size)
        _max_length_chunk_ls = chunk_list(_max_length_ls, _chunk_size)
        return _img_chunk_ls, _task_chunk_ls, _max_length_chunk_ls

    if max_length_ls is None:
        max_length_ls = [None] * len(img_ls)

    img_chunk_ls, task_chunk_ls, max_length_chunk_ls = chunk_inputs(
        img_ls, 
        task_ls, 
        max_length_ls, 
        batch_size
    )

    results = await asyncio.gather(
        *[run_one_batch(
            cur_img_ls, 
            cur_task_ls, 
            cur_max_length_ls
        ) for cur_img_ls, cur_task_ls, cur_max_length_ls in zip(img_chunk_ls, task_chunk_ls, max_length_chunk_ls)], 
        return_exceptions=False
    )

    total_results = flat_nested_list(results)
    repeat_ids = [idx for idx, i in enumerate(total_results) if i.is_repeated]
    has_repeat = len(repeat_ids) > 0
    logger.info(f"has_repeat items: {len(repeat_ids)}, rollback_on_repeat: {rollback_on_repeat}")

    if rollback_on_repeat and has_repeat:
        repeat_img_ls = [img_ls[idx] for idx in repeat_ids]
        repeat_task_ls = [task_ls[idx] for idx in repeat_ids]
        repeat_max_length_ls = [max_length_ls[idx] for idx in repeat_ids]
        repeat_num = len(repeat_ids)

        total_repeat_img_ls = []
        total_repeat_task_ls = []
        total_repeat_max_length_ls = []

        for cur_img, cur_task, cur_max_length in zip(repeat_img_ls, repeat_task_ls, repeat_max_length_ls):
            cur_img_height, cur_img_width = get_img_hw(to_numpy(cur_img))
            nearest_image_shape_ls = get_nearest_image_shape(cur_img_height, cur_img_width, nearest_n=n_nearest)
            assert len(nearest_image_shape_ls) == n_nearest, f"nearest_image_shape_ls must have {n_nearest} elements, but got {len(nearest_image_shape_ls)}"
            
            for _img_height, _img_width in nearest_image_shape_ls:
                img_arr_resized = cv2.resize(to_numpy(cur_img), (_img_width, _img_height))
                total_repeat_img_ls.append(img_arr_resized)
                total_repeat_task_ls.append(cur_task)
                total_repeat_max_length_ls.append(cur_max_length)

        img_repeat_chunk_ls, task_repeat_chunk_ls, max_length_repeat_chunk_ls = chunk_inputs(
            total_repeat_img_ls, 
            total_repeat_task_ls, 
            total_repeat_max_length_ls, 
            batch_size
        )

        repeat_results = await asyncio.gather(
            *[run_one_batch(
                cur_img_ls, 
                cur_task_ls, 
                cur_max_length_ls
            ) for cur_img_ls, cur_task_ls, cur_max_length_ls in zip(img_repeat_chunk_ls, task_repeat_chunk_ls, max_length_repeat_chunk_ls)], 
            return_exceptions=False
        )

        total_repeat_results = flat_nested_list(repeat_results)
        fixed_repeat_num = 0
        
        for i in range(repeat_num):
            start_ids = i * n_nearest
            end_ids = start_ids + n_nearest
            cur_results = total_repeat_results[start_ids:end_ids]
            cur_succeed_results = [i for i in cur_results if i.is_succeed]
            if cur_succeed_results.__len__() == 0:
                continue
                
            non_repeat_results = [i for i in cur_succeed_results if not i.is_repeated]
            if len(non_repeat_results) > 0:
                total_results[repeat_ids[i]] = non_repeat_results[0]
                fixed_repeat_num += 1
        
        logger.info(f"fixed_repeat_num: {fixed_repeat_num}/{repeat_num}")
        return total_results

    return total_results