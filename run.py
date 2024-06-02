from expensifai.image import VisionImageParser

import gradio as gr

from expensifai.expense_report import model
from expensifai.render import render_html
from litellm.caching import Cache
import litellm


def interface():
    demo = gr.Interface(
        fn=processImage,
        inputs=[
            gr.Image(type="filepath"),
            gr.Dropdown(
                ["ollama/llama3", "ollama/phi3"], value="ollama/llama3", label="Model"
            ),
        ],
        outputs=[gr.HTML(), gr.Image(type="filepath")],
    )

    demo.launch()


def processImage(file, model_name):
    observations, image = VisionImageParser.parse(file)
    text = "\n".join([observation.text for observation in observations])
    resp = model(model_name, file, text)
    yield render_html(resp), image


if __name__ == "__main__":
    litellm.cache = Cache(type="disk", disk_cache_dir="/tmp/.litellm_cache")
    interface()
