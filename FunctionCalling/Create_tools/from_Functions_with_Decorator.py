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
### 1) Creating tools from Functions
###
### Tool Creation: Use the @tool decorator to create a tool. 
### A tool is an association between a function and its schema.
### ============================================================================================

from langchain_core.tools import tool

""" Note*:
    - The decorator will use the function's docstring as the tool's description 
    --> so a docstring MUST be provided.
"""
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two members"""
    return a * b


# print(multiply.name)
# print(multiply.description)
# print(multiply.args)


""" Note*
    - @tool supports parsing of annotations, nested schemas, and other features:
"""
from typing import Annotated, List


@tool
def multiply_by_max(
    a: Annotated[int, "scale factor"],
    b: Annotated[List[int], "list of ints over which to take maximum"],
) -> int:
    """Multiply a by the maximum of b."""
    return a * max(b)

# print(multiply_by_max.name)
# print(multiply_by_max.description)
# print(multiply_by_max.args)
# print(multiply_by_max.args_schema.schema())

###
### Customize the tool name and JSON args by passing them into the tool decorator.
###

from pydantic import BaseModel, Field

# Define schema first
class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")
    
@tool("multiplication-tool", args_schema=CalculatorInput, return_direct=True)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# print(multiply.name)
# print(multiply.description)
# print(multiply.args)
# print(multiply.return_direct)
# print(multiply.args_schema.schema())

###
### Use Google Style docstrings
###

""" Note*:
    - @tool can optionally parse Google Style docstrings and 
    associate the docstring components (such as arg descriptions) 
    to the relevant parts of the tool schema
"""

@tool(parse_docstring=True)
def person(name: str, age: int) -> str:
    """The person.

    Args:
        name: The name.
        age: The age.
    """
    return name


# print(person.args_schema.schema())

""" !!! Caution:
    - By default, @tool(parse_docstring=True) will raise ValueError 
    if the docstring does not parse correctly.
"""