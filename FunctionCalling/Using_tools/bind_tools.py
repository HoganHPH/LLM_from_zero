### ============================================================================================
### HOW TO USING TOOLS?

""" 
Docs:
    There are 3 steps:

    1) Tool Binding;
    2) Tool Calling;
    3) Tool Excecution, 
"""

### ============================================================================================
### 1) Tool Binding
### ============================================================================================

###
### Defining tool schemas and Bind to the Chat Model 
### 


"""
Docs:
    4 ways to define tool schemas:
    - Python functions
    - LangChain Tool (@tool decorator)
    - Pydantic class
    - TypedDict class
"""

### Python functions


# The function name, type hints, and docstring are all part of the tool
# schema that's passed to the model. Defining good, descriptive schemas
# is an extension of prompt engineering and is an important part of
# getting models to perform well.
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b


### @tool decorator


### Pydantic class

from pydantic import BaseModel, Field


class add(BaseModel):
    """Add two integers."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class multiply(BaseModel):
    """Multiply two integers."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")
    
    

### TypedDict class
from typing_extensions import Annotated, TypedDict


class add(TypedDict):
    """Add two integers."""

    a: Annotated[int, ..., "First integer"]
    b: Annotated[int, ..., "Second integer"]


class multiply(TypedDict):
    """Multiply two integers."""

    a: Annotated[int, ..., "First integer"]
    b: Annotated[int, ..., "Second integer"]


tools = [add, multiply]

### Define Chat model

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)
chat_model = ChatHuggingFace(llm=llm)


### Bind the tools into the chat model

chat_model_with_tools = chat_model.bind_tools(tools)


### Test
query = "What is 3 * 12?"
response = chat_model_with_tools.invoke(query)
print(response)

"""
Result:
    - LLM generated arguments to a tool
"""