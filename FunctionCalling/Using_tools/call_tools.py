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
### 1) Tool Calling
### ============================================================================================

"""
Docs:
    - If tool calls are included in a LLM response, they are attached to the corresponding 
    message or message chunk as a list of tool call objects in the .tool_calls attribute.
"""

""" Note*:
    - Chat models can call multiple tools at once
"""

from langchain_core.tools import tool

### Define Schema and Tools
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")
    
@tool("multiplication-tool", args_schema=CalculatorInput, return_direct=True)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool("addition-tool", args_schema=CalculatorInput, return_direct=True)
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


tools = [add, multiply]

### Define Chat model

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=1024,
    do_sample=False,
)
chat_model = ChatHuggingFace(llm=llm)

### Bind the tools into the chat model
chat_model_with_tools = chat_model.bind_tools(tools)

### Calling tools
# query = "What is 3 * 12? Also, what is 11 + 49?"
query = "Hello world!"
response = chat_model_with_tools.invoke(query)
print(response)