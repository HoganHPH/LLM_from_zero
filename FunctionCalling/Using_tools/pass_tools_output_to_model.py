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
### 4) Pass tool outputs to Chat model
### Pass the results of the called function back to Chat model so that Chat model can generate
### a message for user.
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
query = "What is 3 multiply 12? Also, what is 11 add 49?"
# query = "Hello world!"


### Execute and Pass tool outputs to Chat model
from langchain_core.messages import HumanMessage

messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)

messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    
    selected_tool = {"addition-tool": add, "multiplication-tool": multiply}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)


""" Note*:
    - ToolMessage must include a tool_call_id that matches an id in the original tool calls 
    that the model generates. This helps the model match tool responses with tool calls.
"""

response = llm_with_tools.invoke(messages)
print(response.content)