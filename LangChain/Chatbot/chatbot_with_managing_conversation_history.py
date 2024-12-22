import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage, trim_messages

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
### STEP 2: Define prompt template
###
prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("<|system|> You are an assistant. Answer all questions to the best of your ability like a {role_play}<|end|>"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

###
### STEP 3: Define an instance containing variables for prompt
###
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    role_play: str


###
### STEP 4: Define trimmer for saving conversation history
###
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=chat_model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
    HumanMessage(content="whats 3 + 2"),
    AIMessage(content="5"),
    HumanMessage(content="whats 2 + 4"),
    AIMessage(content="6"),
    HumanMessage(content="whats 2 + 5"),
    AIMessage(content="7"),
    HumanMessage(content="whats 2 + 6"),
    AIMessage(content="8"),
    HumanMessage(content="whats 2 + 7"),
    AIMessage(content="9"),
    HumanMessage(content="whats 2 + 8"),
    AIMessage(content="10"),
]

trimmer.invoke(messages)

###
### STEP 5: Define workflow of LangGraph
###
workflow = StateGraph(state_schema=State)

def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "role_play": state["role_play"]}
    )
    response = chat_model.invoke(prompt)
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

###
### STEP 6: Define config and Invoke message
###

# 5.1. The information from earliest conversation is trimed (forget)
config = {"configurable": {"thread_id": "user_1234"}}
query = "What is my name?"
role_play = "pirate"

input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "role_play": role_play},
    config,
)
output["messages"][-1].pretty_print()

# 5.2. Still remember the last few messages
config = {"configurable": {"thread_id": "abc678"}}
query = "What math problem did I ask?"
role_play = "pirate"

input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "role_play": role_play},
    config,
)
output["messages"][-1].pretty_print()