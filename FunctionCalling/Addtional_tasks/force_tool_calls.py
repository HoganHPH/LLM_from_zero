### ===================================================================================
### ===================================================================================
###
### How to force models to call a tool
###
### ===================================================================================
### ===================================================================================

"""
Docs:
    - In order to force our LLM to select a specific tool, 
    we can use the tool_choice parameter to ensure certain behavior
"""

###
### Define tools
###

from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

### Optional: convert to json format of OpenAI
# from langchain_core.utils.function_calling import convert_to_openai_tool
# add = convert_to_openai_tool(add)
# multiply = convert_to_openai_tool(multiply)
tools = [add, multiply]

###
### Define Chat model
###

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


###
### Force tool to call
###
llm_forced_to_multiply = llm.bind_tools(tools, tool_choice="multiply")
response = llm_forced_to_multiply.invoke("what is 2 + 4")
print(response)