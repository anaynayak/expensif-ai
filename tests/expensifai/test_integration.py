from pathlib import Path
from expensifai.clustering import cluster
from expensifai.expense_report import model
from expensifai.vision import detect_text

image_path = Path(__file__).parent / "food.jpg"


def test_should_return_correct_output(llm_cache):
    path = str(image_path.absolute())
    observations, _ = detect_text(path)
    observations, _ = cluster(path, observations)
    vision_text = "\n".join([observation.text for observation in observations])

    expense_items = model("ollama/llama3", image_path, vision_text)

    assert expense_items == {
        "items": [
            {
                "amount": 449.0,
                "category": "Food",
                "date": "25 Apr'24",
                "name": "Take-Out 1 Croissant Avocado",
                "quantity": 1,
                "action": "REVIEW",
            },
        ]
    }
