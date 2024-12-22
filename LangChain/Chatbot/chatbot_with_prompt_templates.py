import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate

from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


###
### STEP 1: Define a LLM ChatModel
### Here I use a HunggingFace checkpoint to create model
###
llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)
chat_model = ChatHuggingFace(llm=llm)
# print(chat_model)


###
### STEP 2: Define a prompt template
### Note*: The template is based on each model when using HuggingFace
###

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You talk like a baby. Answer all questions to the best of your ability."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


###
### STEP 3: Define chatbot app using memory workflow of LangGraph
###
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = chat_model.invoke(prompt)
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


###
### STEP 4: Define config and Use defined chatbot app to invoke human message 
###

# 4.1.
config = {"configurable": {"thread_id": "user_1"}}
query = "Hi! My name is Jim."
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

# 4.2.
config = {"configurable": {"thread_id": "user_1"}}
query = "What is my name?"
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()


###
### MORE COMPLEXITY PROMPT: prompt with multiple input (more instructions)
###

# STEP 1: Define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("<|system|> You talk like a baby. Answer all questions to the best of your ability in {language}<|end|>"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# STEP 2: Define an instance containing variables for prompt
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

workflow = StateGraph(state_schema=State)

# STEP 3: Init workflow of LangGraph
def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = chat_model.invoke(prompt)
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# STEP 4: Pass into instructions for prompt and Invoke message

# 4.1.
config = {"configurable": {"thread_id": "user_1"}}
query = "Hi! I'm Bob."
language = "Spanish"

input_messages = [HumanMessage(query)]
output = app.invoke(
    {"language": language, "messages": input_messages},
    config,
)
output["messages"][-1].pretty_print()

# 4.2.
query = "What is my name?"
input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages},
    config,
)
output["messages"][-1].pretty_print()