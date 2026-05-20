import os 
import re
import numpy as np
import cv2
from PIL import Image
from moss_ocr.inferer.vllm.basic_runner import BaseVllmPipelineOpenAI, OCRResult
from moss_ocr.utils.utils import to_numpy, ndarray2base64str_cv2, to_numpy
import logging
from typing import Literal, Sequence, Union, Any

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


class MOSSOCRvLLMRunner(BaseVllmPipelineOpenAI):
    TASK_PROMPT_MAP = dict(
        math="read formula from image and output in Latex formula format: \n",
        table="read table from image and output in HTML format: \n",
        text="read text from image and output in Markdown format: \n",
    )

    def postprocessing(self, result, task, **kwargs) -> Any:
        if task == "table":
            result = result.replace("<html><body>", "").replace("</body></html>", "")
            if result.startswith("<table>") and not result.endswith("</table>"):
                result = result + "</table>"

            return result
        if task == "math":
            result = result.strip("$").strip()
            result = result.replace(r"\[", "")
            result = result.replace(r"\]", "")
            result = f"$$\n{result}\n$$"
            return result
        else:
            return super().postprocessing(result, task)

    def build_payload(
        self,
        img: Union[np.ndarray, str, Image.Image],
        task: str,
        max_length: int | None = None,
        **kwargs
    ):
        img_arr = to_numpy(img)
        img_base64 = ndarray2base64str_cv2(img_arr)
        url  = f"data:image/png;base64,{img_base64}"
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": url
                            }
                        },
                        {
                            "type": "text",
                            "text": self.TASK_PROMPT_MAP[task]
                        }
                    ]
                }
            ],
            "max_completion_tokens": max_length or self.max_length,
            "temperature": 0,
            "top_p": 1.0, 
        }
        return payload


class MOSSOCRv1d6vLLMRunner(MOSSOCRvLLMRunner):
    MODEL_NAME = "moss-v1d6-0.3b"
    MAX_LENGTH = 2048
