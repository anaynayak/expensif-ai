from pathlib import Path
from datasets import load_dataset
from model.expense_report import model
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

    vision_text = detect_text(image)
    expense_items = model("ollama/llama3", image, vision_text)

    pp([gt, vision_text, expense_items])

    assert expense_items[0].name == menu[0]["nm"]
    assert expense_items[0].quantity == menu[0]["cnt"]
    assert expense_items[0].amount == menu[0]["price"]
    assert expense_items[0].action == "REVIEW"
    assert expense_items[0].category == "Food"
