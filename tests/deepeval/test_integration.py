from pathlib import Path
from model.expense_report import model
from parser.vision import detect_text

image_path = Path(__file__).parent.parent / "parser" / "food.jpg"


def test_should_return_correct_output(llm_cache):
    path = str(image_path.absolute())

    vision_text = [observation.text for observation in detect_text(path)]

    expense_items = model("ollama/llama3", image_path, vision_text)

    assert expense_items == {
        "items": [
            {
                "date": "",
                "name": "Take-Out",
                "quantity": 1,
                "amount": 449.0,
                "category": "Food",
                "action": "REVIEW",
            },
            {
                "date": "",
                "name": "Croissant Avocado",
                "quantity": 1,
                "amount": 471.5,
                "category": "Food",
                "action": "REVIEW",
            },
        ]
    }
