### =====================================================================================
### =====================================================================================
### HOW TO : STREAM RUNNABLES
### =====================================================================================
### =====================================================================================

"""
Docs:
    - 2 general approaches to stream content:
        +) sync "stream" and async "astream": stream final output of the chain
        +) async "astream_events" and async "astream_log": stream both intermediate steps
        and final output from the chain
"""

###
### Using Stream
###

### Define Chat model
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# llm = HuggingFaceEndpoint(
#     repo_id="microsoft/Phi-3-mini-4k-instruct",
#     task="text-generation",
#     max_new_tokens=100,
#     do_sample=False,
# )
# chat_model = ChatHuggingFace(llm=llm)

import os
from dotenv import load_dotenv
load_dotenv()

### OpenAI
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI
chat_model = ChatOpenAI(model="gpt-4o-mini")

###
### sync "stream" API 
###
# chunks = []
# for chunk in chat_model.stream("what color is the sky?"):
#     chunks.append(chunk)
#     print(chunk.content, end="|", flush=True)

###
### Chains
###
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
parser = StrOutputParser()
chain = prompt | chat_model | parser

for chunk in chain.stream({"topic": "parrot"}):
    print(chunk, end="|", flush=True)