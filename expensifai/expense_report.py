from typing import List, Optional
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_core.prompts import PromptTemplate
from langfuse.callback import CallbackHandler
from langchain.output_parsers import OutputFixingParser


class ExpenseItem(BaseModel):
    date: Optional[str] = Field(description="Date of the expense in yyyy-mm-dd format")
    name: str = Field(description="Name of the item")
    quantity: Optional[int] = Field(description="Quantity of the item")
    amount: Optional[float] = Field(description="Amount of the item")
    category: Optional[str] = Field(description="Category of the item")
    action: Optional[str] = Field(description="Action to be taken on the item")


class ExpenseItems(BaseModel):
    items: List[ExpenseItem] = Field(description="List of all expense items")


def model(model_name: str, file: str, query: str, llm_ops: bool) -> ExpenseItems:
    model = ChatLiteLLM(model=model_name, temperature=0, max_tokens=1000)

    parser = SimpleJsonOutputParser(pydantic_object=ExpenseItems)
    prompt = PromptTemplate(
        template="""You are a member of the finance team responsible for reviewing all submitted expenses.
            You have received the image "{file}" from a colleague.
            The following items are not permitted as per the policy and should have the `action` marked as `FLAGGED`:
                1. Alcoholic drinks
                2. Cigarettes
                3. Personal items
                4. Drinks
            The following items are permitted and should have the `action` marked as `APPROVE`:
                1. Transport/Cab expenses
                2. Food expenses
                3. Hotel bookings
            {format_instructions}
            Only respond with the items that need to be flagged, approved, or reviewed. Do not add any additional explanation.
            {query}
            """,
        input_variables=["file", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    new_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
    chain = prompt | model | new_parser
    callbacks = [CallbackHandler()] if llm_ops else []
    resp = chain.invoke({"query": query, "file": file}, config={"callbacks": callbacks})
    return ExpenseItems.parse_obj(resp)
