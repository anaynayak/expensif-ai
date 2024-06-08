from pathlib import Path
from datasets import load_dataset
import json
import pytest

from run import processImage

images_path = Path(__file__).parent / "images"


dataset = load_dataset("naver-clova-ix/cord-v2", split="train[:1%]")


@pytest.mark.parametrize("idx, data", enumerate(dataset))
def test_should_return_correct_output(idx, data, llm_cache):
    image = str((images_path / f"test{idx}.png").absolute())
    data["image"].save(image)
    gt = json.loads(data["ground_truth"])
    gt["gt_parse"]["menu"]

    expense_items, _ = processImage(image, "ollama/llama3", False)
    assert len(expense_items.items) > 0
