from typing import Generator, List, Tuple
from expensifai.image import VisionImageParser
import argparse
import gradio as gr

from expensifai.expense_report import ExpenseItems, model
from expensifai.render import render_html
from litellm.caching import Cache
from PIL.Image import Image
import litellm


def interface(models: List[str]):
    demo = gr.Interface(
        fn=process,
        analytics_enabled=False,
        inputs=[
            gr.Image(type="filepath"),
            gr.Dropdown(models, value=models[0], label="Model"),
            gr.Checkbox(label="LLM Ops", value=False),
        ],
        outputs=[gr.HTML(), gr.Image(type="filepath")],
    )

    demo.launch()


def processImage(file, model_name, llm_ops) -> Tuple[ExpenseItems, Image]:
    observations, image = VisionImageParser.parse(file)
    text = "\n".join([observation.text for observation in observations])
    return model(model_name, file, text, llm_ops), image


def process(file, model_name, llm_ops) -> Generator[Tuple[str, Image], None, None]:
    resp, image = processImage(file, model_name, llm_ops)
    yield render_html(resp), image


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=lambda s: s.split(","),
        default=["ollama/llama3", "ollama/phi3"],
        help="Model to use for processing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    litellm.cache = Cache(type="disk", disk_cache_dir="/tmp/.litellm_cache")
    args = arg_parse()
    interface(args.models)
