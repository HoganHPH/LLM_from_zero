import os

from dotenv import load_dotenv
from typing import Optional, List
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
# Setup LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


###
### SCHEMA: describe what information we want to extract from the text
###
class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )
    
""" **Notes** : What do I need to do when defining schema?
(1) Define exactly ATTRIBUTES that are the information LLM need to return 
(2) DO NOT force LLM make up information (that means fake information)
"""

###
### PROMPT: instructions for extractor (LLM model) in a specific form
### Details of template for this model: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
###

prompt_template = ChatPromptTemplate.from_template(
    """
        <|system|>
        You are an expert extraction algorithm.<|end|>
        <|user|>
        Only extract relevant information from the text.
        If there is no value of an attribute to extract, return null for the attribute's value, do not generate.
        Passage:
        {input}
        <|end|>
        <|assistant|>
    """
)

###
### MODEL: 
### +) using pretrained from HuggingFace, 
### +) bind tool calling (schema) and provide with prompt
###

# Use HuggingFaceEndpoint and ChatHunggingFace to define 
# an instance LLM model in LangChain using HuggingFace pretrained checkpoint
llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)
chat_model = ChatHuggingFace(llm=llm)


###
### FUNCTION CALLING
###

structured_llm = chat_model.bind_tools([Person])
# print(structured_llm)

###
### EXECUTION: invoke and result
###
input = "Alan Smith is 1.75 meters tall and has brown hair."
prompt = prompt_template.invoke({"input": input})
# print(prompt)

response = structured_llm.invoke(prompt).dict()
print(response['tool_calls'])


###
### Multiple Entities
###

class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]


structured_llm = chat_model.bind_tools([Data])
input = "My name is Jeff, my hair is black and i am 1.65 meters tall. Anna is my friend, she is 1.6m and she has a yellow hair."
prompt = prompt_template.invoke({"input": input})
# print(prompt)

response = structured_llm.invoke(prompt).dict()
print(response['tool_calls'])


###
### Reference examples
###
messages = [
    {"role": "user", "content": "2 + 2"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "2 + 3"},
    {"role": "assistant", "content": "5"},
    {"role": "user", "content": "3 + 4"},
]

response = llm.invoke(messages)
print(response)













