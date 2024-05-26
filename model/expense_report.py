from typing import List, Optional
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langfuse.callback import CallbackHandler


class ExpenseItem(BaseModel):
    date: Optional[str] = Field(description="Date of the expense in yyyy-mm-dd format")
    name: str = Field(description="Name of the item")
    quantity: Optional[int] = Field(description="Quantity of the item")
    amount: Optional[float] = Field(description="Amount of the item")
    category: Optional[str] = Field(description="Category of the item")
    action: Optional[str] = Field(description="Action to be taken on the item")


class ExpenseItems(BaseModel):
    items: List[ExpenseItem] = Field(description="List of expense items")


def model(model_name: str, file: str, query: str) -> str:
    model = ChatLiteLLM(model=model_name)

    parser = PydanticOutputParser(pydantic_object=ExpenseItems)
    prompt = PromptTemplate(
        template="""You are a member of the finance team responsible for reviewing all submitted expenses.
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
            5. Travel expenses
            {format_instructions}
            {query}
            """,
        input_variables=["file", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser
    langfuse_handler = CallbackHandler()
    return chain.invoke(
        {"query": query, "file": file}, config={"callbacks": [langfuse_handler]}
    )
