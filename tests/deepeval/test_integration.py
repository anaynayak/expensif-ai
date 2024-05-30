from pathlib import Path
from model.expense_report import model
from parser.vision import detect_text
from pprint import pprint

image_path = Path(__file__).parent.parent / "parser" / "food.jpg"


def test_should_return_correct_output(llm_cache):
    path = str(image_path.absolute())

    vision_text = [text for (text, confidence, bbox) in detect_text(path)]

    expense_items = model("ollama/llama3", image_path, vision_text)

    pprint([vision_text, expense_items])
