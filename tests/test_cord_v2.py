from pathlib import Path
from datasets import load_dataset
from model.expense_report import model
from model.clustering import cluster
from parser.vision import detect_text
import json
import pytest

images_path = Path(__file__).parent / "images"

from pprint import pprint as pp

dataset = load_dataset("naver-clova-ix/cord-v2", split="train[:1%]")


@pytest.mark.parametrize("idx, data", enumerate(dataset))
def test_should_return_correct_output(idx, data, llm_cache):
    image = str((images_path / f"test{idx}.png").absolute())
    data["image"].save(image)
    gt = json.loads(data["ground_truth"])
    menu = gt["gt_parse"]["menu"]

    vision_text = cluster(image, detect_text(image))
    expense_items = model("ollama/llama3", image, vision_text)

    pp(["vision text", vision_text])
    pp(["expense items", expense_items])
    pp(["gt", menu])
