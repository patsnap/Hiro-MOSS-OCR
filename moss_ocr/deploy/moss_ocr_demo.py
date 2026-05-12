import os
import logging
import argparse
import numpy as np
import json
import types
from enum import Enum
import gradio as gr
import uuid

import torch
from PIL import Image
from moss_ocr.inferer.cuda_graph.moss_v1d6.moss_v1d6_runner import MOSSv1d6Runner
from moss_ocr.utils.utils import Timer, get_all_file_path, checkdir

from moss_ocr.utils.draw_operation import DrawImage

curdir = os.path.dirname(__file__)
rtpath = os.path.abspath(os.path.join(curdir, "../.."))
gradio_tmp_dir = os.path.join(rtpath, ".cache", "gradio")
checkdir([gradio_tmp_dir])
os.environ['GRADIO_TEMP_DIR'] = gradio_tmp_dir

MAXIMUM_CACHED = {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)


class TaskType(Enum):
    MATH_OCR = "公式OCR"
    TABLE_OCR = "表格OCR"
    TEXT_OCR = "段落文本OCR"


    @classmethod
    def supported_tasks(cls) -> list[str]:
        return [i.value for i in cls]


class Example4Gr:

    @classmethod
    def vllm_example(cls, img_gr: gr.components.Component, task_gr: gr.components.Component) -> gr.Examples:
        mathocr_example_ls = [[i, TaskType.MATH_OCR.value] for i in get_all_file_path(os.path.join(rtpath, "static/img_examples/math"))[:2]]
        table_example_ls = [[i, TaskType.TABLE_OCR.value] for i in get_all_file_path(os.path.join(rtpath, "static/img_examples/table"))[:2]]
        text_example_ls = [[i, TaskType.TEXT_OCR.value] for i in get_all_file_path(os.path.join(rtpath, "static/img_examples/text"))[:2]]

        total_example = mathocr_example_ls + table_example_ls + text_example_ls 
        return gr.Examples(
            examples=total_example,
            inputs=[img_gr, task_gr],
        )


# -------- init model
doc_vllm_obj: MOSSv1d6Runner | None = None
# --------------------------------


def init_model(
    model_path: str, 
    max_length=2048, 
    dtype=torch.bfloat16, 
    max_batch_size: int = 8, 
):
    global doc_vllm_obj
    doc_vllm_obj = MOSSv1d6Runner(
        model_path=model_path,
        max_length=max_length,
        max_batch_size=max_batch_size,
        dtype=dtype,
        logger=logger
    )
    
    print(f"init model {model_path} successfully!, max_length: {max_length}, max_batch_size: {max_batch_size}")


def draw_result_image(
        self,
        img: np.ndarray,
        draw_conf=True,
        target_label: list[str] | str | None = None
) -> np.ndarray:
    target_label: list = self.get_target_label(target_label)
    target_label_set = set(target_label)
    polygon_ls = []
    conf_ls = []
    label_ls = []

    for polygon, label, conf in zip(self.polygon_ls, self.label_ls, self.conf_ls):
        if label in target_label_set:
            polygon_ls.append(polygon)
            conf_ls.append(conf)
            label_ls.append(label)
    if len(target_label) > 0:

        return DrawImage.poly_with_annotation_new(
            img=img,
            boxes=polygon_ls,
            labels=[
                f"[{idx:02d}]{cur_label}({cur_conf:.4f})" if draw_conf else f"[{idx:02d}]{cur_label}"
                for idx, (cur_label, cur_conf) in enumerate(zip(label_ls, conf_ls))
            ],
            color=DrawImage.generate_color_seq(label_ls),
            draw_vertical_text=False
        )
    else:
        return img


class OCRGrBackend:

    @classmethod
    def math_ocr_streaming(cls, _img: np.ndarray) -> list:
        timer = Timer()
        latex = ""
        token_ids_ls = []
        for cur_token_ids in doc_vllm_obj.run_streaming(_img, task="math"):
            token_ids_ls.append(cur_token_ids)
            latex += doc_vllm_obj.tokenizer.batch_decode(cur_token_ids, skip_special_tokens=True)[0]
            yield latex, latex, None

        timer.record("mathocr")
        if not latex.startswith("$$"):
            latex = f"$$\n{latex}\n$$"
        logger.info(f"\n{timer.summary(return_type='table')}")
        yield [latex, latex, timer.summary(return_type="html")]

    @classmethod
    def math_ocr(cls, _img: np.ndarray) -> list:
        timer = Timer()
        latex = doc_vllm_obj.run(_img, task="math")
        timer.record("mathocr")
        if not latex.startswith("$$"):
            latex = f"$$\n{latex}\n$$"
        logger.info(f"\n{timer.summary(return_type='table')}")
        return [latex, latex, timer.summary(return_type="html")]

    @classmethod
    def table_ocr(cls, _img: np.ndarray) -> list:
        timer = Timer()
        table_html = doc_vllm_obj.run(_img, task="table")
        timer.record("table_ocr")
        logger.info(f"\n{timer.summary(return_type='table')}")
        return [table_html, table_html, timer.summary(return_type="html")]

    @classmethod
    def text_ocr(cls, _img: np.ndarray, task: str = "text") -> list:
        timer = Timer()
        text = doc_vllm_obj.run(_img, task=task)
        timer.record("text_ocr")
        logger.info(f"\n{timer.summary(return_type='table')}")
        return [text, text, timer.summary(return_type="html")]

    @classmethod
    def table_ocr_streaming(cls, _img: np.ndarray) -> list:
        timer = Timer()
        table_html = ""
        token_ids_ls = []
        for cur_token_ids in doc_vllm_obj.run_streaming(_img, task="table"):
            token_ids_ls.append(cur_token_ids)
            table_html += doc_vllm_obj.tokenizer.batch_decode(cur_token_ids, skip_special_tokens=True)[0]
            yield table_html, table_html, None, None
        timer.record("table_ocr")
        token_ids = torch.cat(token_ids_ls, dim=1)
        table_html = doc_vllm_obj.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
        logger.info(f"\n{timer.summary(return_type='table')}")
        yield [
            gr.update(elem_id="vllm_markdown_result", value=table_html),
            gr.update(elem_id="vllm_markdown_source_result", value=table_html),
            gr.update(elem_id="vllm_time_info", visible=False, value=timer.summary(return_type="html")),
        ]

    @classmethod
    def text_ocr_streaming(cls, _img: np.ndarray, task: str = "text") -> list:
        timer = Timer()
        text = ""
        token_ids_ls = []
        for cur_token_ids in doc_vllm_obj.run_streaming(_img, task=task):
            token_ids_ls.append(cur_token_ids)
            text += doc_vllm_obj.tokenizer.batch_decode(cur_token_ids, skip_special_tokens=True)[0]
            yield text, text,  None, None
        timer.record("text_ocr")
        token_ids = torch.cat(token_ids_ls, dim=1)
        text = doc_vllm_obj.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
        logger.info(f"\n{timer.summary(return_type='table')}")
        yield text, text, timer.summary(return_type="html")

    @classmethod
    def moss_ocr(cls, _img: np.ndarray, task: str):
        handler = {
            TaskType.MATH_OCR.value: cls.math_ocr if not hasattr(doc_vllm_obj, "run_streaming") else cls.math_ocr_streaming,
            TaskType.TABLE_OCR.value: cls.table_ocr if not hasattr(doc_vllm_obj, "run_streaming") else cls.table_ocr_streaming,
            TaskType.TEXT_OCR.value: cls.text_ocr if not hasattr(doc_vllm_obj, "run_streaming") else cls.text_ocr_streaming,
        }[task]

        res = handler(_img)

        if isinstance(res, types.GeneratorType):
            # streaming
            yield from res
        else:
            # not streaming
            yield res


class OCR4GrFrontend:

    @classmethod
    def doc_vllm_gr(cls):
        with gr.Row():
            with gr.Column():
                with gr.Tab("Image"):
                    upload_vllm_image = gr.Image(
                        label='待识别图片', type="numpy", image_mode='RGB', elem_id="upload_vllm_image"
                    )

                task_radio = gr.Radio(
                    label="任务选择",
                    choices=TaskType.supported_tasks(),
                    value=TaskType.PAGE_OCR.value
                )

            with gr.Column():
                with gr.Tab(label="Markdown渲染结果"):
                    vllm_markdown_result = gr.Markdown(
                        label="vllm_markdown_result",
                        elem_id="vllm_markdown_result",
                        latex_delimiters=[
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": "$", "right": "$", "display": False},
                            {"left": r"\[", "right": r"\]", "display": True},
                            {"left": r"\(", "right": r"\)", "display": False}
                        ]
                    )
                with gr.Tab(label="原始文本"):
                    vllm_markdown_source_result = gr.Textbox(
                        label="vllm_markdown_source_result",
                        elem_id="vllm_markdown_source_result",
                        show_label=True,
                        max_lines=500,
                        lines=38,
                    )

        with gr.Row():
            vllm_submit_btn = gr.Button("Submit", elem_id="vllm_submit_btn")
        vllm_time_info = gr.HTML(label="耗时统计", elem_id="vllm_time_info", show_label=True)

        with gr.Accordion(label="展开查看Examples", elem_id="vllm_examples", open=False):
            Example4Gr.vllm_example(upload_vllm_image, task_radio)

        vllm_submit_btn.click(
            OCRGrBackend.moss_ocr,
            inputs=[upload_vllm_image, task_radio],
            outputs=[
                vllm_markdown_result,
                vllm_markdown_source_result,
                vllm_time_info,
            ]
        )

    @classmethod
    def ocr_hub_gr(cls, model_name: str = "moss_v1_0_3B_251201"):
        with gr.Blocks(theme=gr.themes.Monochrome()) as hub_demo:
            gr.components.Markdown(
                f"""
                # 🍀OCR-Anything-VLLM ({model_name})
                **MOSS**: **M**ultimodal **O**CR for **S**tructured Markup **S**equencing
                """
            )
            #
            with gr.Tab("多模态 OCR"):
                cls.doc_vllm_gr()

        return hub_demo


def main():
    parser = argparse.ArgumentParser("OCR-Anything")
    parser.add_argument("--port", default=7788, type=int, help="web demo service port")
    parser.add_argument("--max_length", default=2048, type=int, help="generation max length")
    parser.add_argument("--max_batch_size", default=8, type=int, help="max batch size")
    parser.add_argument("--model_name", default="moss_v1_0_3B_251201", choices=MODELHUB.keys(), type=str, help="model name")
    args = parser.parse_args()
    init_model(max_length=args.max_length, max_batch_size=args.max_batch_size, model_name=args.model_name)

    demo = OCR4GrFrontend().ocr_hub_gr(model_name=MODLE_COFNIG[args.model_name].model_name_display)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        debug=True,
        allowed_paths=[rtpath, curdir]
    )


if __name__ == "__main__":
    main()
