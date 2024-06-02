from parser.image import VisionImageParser

import gradio as gr

from model.expense_report import model
from render import render_html


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
    observations = VisionImageParser.parse(file)
    text = "\n".join([observation.text for observation in observations])
    resp = model(model_name, file, text)
    yield render_html(resp), "/tmp/cluster.png"


if __name__ == "__main__":
    interface()
