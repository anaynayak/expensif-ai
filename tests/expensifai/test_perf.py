from pathlib import Path
from typing import Tuple
from expensifai.expense_report import ExpenseItem, ExpenseItems
from run import processImage
from PIL.Image import Image

image_path = Path(__file__).parent / "food.jpg"


def run() -> Tuple[ExpenseItems, Image]:
    path = str(image_path.absolute())
    return processImage(path, "ollama/llama3", False)


def test_benchmark(benchmark):
    items, _ = benchmark(run)
    assert items == ExpenseItems(
        items=[
            ExpenseItem(
                amount=449.0,
                category="Food expenses",
                date="25 Apr'24 19:31 PM",
                name="Croissant Avocado",
                quantity=1,
                action="APPROVE",
            )
        ]
    )
