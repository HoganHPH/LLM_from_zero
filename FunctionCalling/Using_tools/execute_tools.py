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
### 3) Tool Execution
### Execute the tool function and return results of the called function
### ============================================================================================


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
# print(tools)

### Define Chat model

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# from langchain_mistralai import ChatMistralAI

# llm = ChatMistralAI(model="mistral-large-latest")

# response = llm.invoke("Hello, my name is Hoang")
# print(response)

### Bind the tools into the chat model
llm_with_tools = llm.bind_tools(tools)

### Calling tools
query = "What is 3 * 12? Also, what is 11 + 49?"
# query = "Hello world!"

response = llm_with_tools.invoke(query)

returned_tools = response.tool_calls
if len(returned_tools) > 0:
    for tool in returned_tools:
        # print(tool['name'])
        args = tool['args']
        # print(args)
        matching_tool= [t for t in tools if t.name == tool['name']]
        response = matching_tool[0].invoke(args)
        print(response)
else:
    print("No tool call")