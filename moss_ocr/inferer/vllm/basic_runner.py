import os
import logging
import re
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
from moss_ocr.utils.utils import truncate_repetitions_fast_slice


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
        url: str | None = None,
        api_key: str | None = "EMPTY",
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

        self.logger = logger or _logger

        self._current_loop = None
        self._http_client = None
        self._async_client = None
        self.detect_repeat = detect_repeat
        self._url_index = 0
        self._async_clients = []
        self.ocr_sem = asyncio.Semaphore(max_concurrent)

    def _get_client(self) -> AsyncOpenAI:
        loop = asyncio.get_running_loop()
        
        if self._current_loop is not loop:
            self._current_loop = loop
            self._url_index = 0
            old_client = self._http_client
            
            limits = httpx.Limits(
                max_connections=self.max_concurrent + 10,
                max_keepalive_connections=self.max_concurrent
            )
            self._http_client = httpx.AsyncClient(limits=limits, timeout=self.timeout)
            
            if old_client and getattr(old_client, "is_closed", False) is False:
                try:
                    loop.create_task(old_client.aclose())
                except Exception as e:
                    self.logger.warning(f"Failed to close old http client: {e}")

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
                            None, 
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

        async def bounded_run(img, task, length):
            async with self.ocr_sem:
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
        if self._http_client and not self._http_client.is_closed:
            try:
                asyncio.run(self._http_client.aclose())
            except Exception:
                pass

    def __del__(self):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    async def aclose(self):
        """close the http client"""
        if self._http_client and getattr(self._http_client, "is_closed", False) is False:
            await self._http_client.aclose()
