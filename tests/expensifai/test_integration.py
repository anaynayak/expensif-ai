from pathlib import Path
from expensifai.expense_report import ExpenseItem, ExpenseItems
from run import processImage

image_path = Path(__file__).parent / "food.jpg"


def test_should_return_correct_output(llm_cache):
    path = str(image_path.absolute())
    expense_items, _ = processImage(path, "ollama/llama3", False)

    assert expense_items == ExpenseItems(
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
