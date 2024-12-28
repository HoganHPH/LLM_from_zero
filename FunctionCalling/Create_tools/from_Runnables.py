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
### 2) Creating tools from Runnables
### ============================================================================================

"""
Docs:
    - "as_tool" method
    - specification of names, descriptions, and additional schema information for arguments.
"""

from langchain_core.language_models import GenericFakeChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [("human", "Hello. Please respond in the style of {answer_style}.")]
)

# Placeholder LLM
llm = GenericFakeChatModel(messages=iter(["hello matey"]))

chain = prompt | llm | StrOutputParser()

as_tool = chain.as_tool(
    name="Style responder", description="Description of when to use tool."
)
print(as_tool.args)