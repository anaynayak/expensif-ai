from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage


def model(model: str, file: str, text: str) -> str:
    chat = ChatLiteLLM(model=model)
    response = chat.invoke(
        [
            SystemMessage(
                content=f"""
            You are a member of the finance team responsible for reviewing all submitted expenses.
            You have received the image "{file}" from a colleague.
            The following items are not permitted as per the policy and need to be flagged.
            1. Hard liquor
            2. Cigarettes
            3. Personal items
            The following items are permitted but need to be reviewed.
            1. Cab visits to the airport/office
            2. Meals with clients
            3. Hotel stays
            4. Office supplies
            Only print the report in a markdown format, with the following columns:
            1. Date (yyyy-mm-dd)
            2. Item name
            3. Quantity
            4. Amount
            5. Category
            6. Action (review/flag/approve)
            7. Inline Image
            Do not include the system message in your response.
            Do not add anything else to the report.
            """,
            ),
            HumanMessage(content=text),
        ]
    )
    return response.content
