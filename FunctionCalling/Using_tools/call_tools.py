### ============================================================================================
### HOW TO USING TOOLS?

""" 
Docs:
    There are 3 steps:

    1) Tool Binding;
    2) Tool Calling;
    3) Tool Excecution;
    4) Pass tool outputs to Chat model;
"""

### ============================================================================================
### 2) Tool Calling
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
    
@tool("multiplication-tool", args_schema=CalculatorInput, return_direct=False)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool("addition-tool", args_schema=CalculatorInput, return_direct=False)
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


tools = [multiply, add]
print(tools)

### Define Chat model

import os
from dotenv import load_dotenv
load_dotenv()

### OpenAI
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

### MistralAI
# from langchain_mistralai import ChatMistralAI
# llm = ChatMistralAI(model="mistral-large-latest")


### Bind the tools into the chat model
llm_with_tools = llm.bind_tools(tools)

### Calling tools
query = "What is 3 * 12? Also, what is 11 + 49?"
# query = "Hello world!"

response = llm_with_tools.invoke(query)

print(response)
print()
print(response.tool_calls)