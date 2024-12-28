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
### 1) Creating tools from Function
###
### The StructuredTool.from_function class method provides 
### a bit more configurability than the @tool decorator
### ============================================================================================

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


### Without configuration
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(func=multiply)

# print(calculator.invoke({"a": 2, "b": 3}))


### With configuration
class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(
    func=multiply,
    name="Calculator",
    description="multiply numbers",
    args_schema=CalculatorInput,
    return_direct=True,
    # coroutine= ... <- you can specify an async method if desired as well
)

print(calculator.invoke({"a": 100, "b": 100}))
print(calculator.name)
print(calculator.description)
print(calculator.args)