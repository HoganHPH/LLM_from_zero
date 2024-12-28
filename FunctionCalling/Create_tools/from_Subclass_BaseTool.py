### ============================================================================================
### HOW TO CREATE TOOLS?

""" 
Docs:
    LangChain supports the creation of tools from:

    1) Functions;
    2) LangChain Runnables;
    3) By sub-classing from BaseTool -- This is the most flexible method, 
    it provides the largest degree of control, at the expense of more effort and code.
    
Note*:
- Models will perform better if the tools have well chosen names, descriptions and JSON schemas.
"""

### ============================================================================================
### 3) Creating tools by Subclass BaseTool
###
### Provides maximal control over the tool definition, but requires writing more code.
### ============================================================================================

from typing import Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")
    
# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class CustomCalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(
        self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return a * b

multiply = CustomCalculatorTool()
print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.return_direct)

print(multiply.invoke({"a": 2, "b": 3}))