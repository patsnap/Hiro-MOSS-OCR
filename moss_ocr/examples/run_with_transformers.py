import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from transformers import AutoModelForCausalLM


def run_demo(model_path: str, task: str, img_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    ).eval()

    with torch.inference_mode():
        texts = model.generate(img_path, task=task)

    print(texts[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    args = parser.parse_args()
    run_demo(args.model_path, args.task, args.img_path)
