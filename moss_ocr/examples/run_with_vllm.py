import argparse
from moss_ocr.inferer.vllm import MOSSOCRv1d6vLLMRunner


def run_demo(task, img_path: str, url: str):
    runner = MOSSOCRv1d6vLLMRunner(url=url)
    result = runner.run(img=img_path, task= task)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--url", type=str, required=True)
    args = parser.parse_args()
    run_demo(args.task, args.img_path, args.url)
