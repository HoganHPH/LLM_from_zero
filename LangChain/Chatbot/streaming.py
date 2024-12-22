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

###
### STEP 2: Define a prompt template
### Note*: The template is based on each model when using HuggingFace
###
prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("<|system|> You talk like a baby. Answer all questions to the best of your ability in {language}<|end|>"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

###
### STEP 3: Define an instance containing variables for prompt
###
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

workflow = StateGraph(state_schema=State)

###
### STEP 4: Init workflow of LangGraph
###
def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = chat_model.invoke(prompt)
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


###
### STEP 5: Stream back each token as it is generated
### Note*: Now, chunk return a full sentence, not characters
###
config = {"configurable": {"thread_id": "abc789"}}
query = "Hi I'm Todd, please tell me a joke."
language = "Spanish"

input_messages = [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"language": language, "messages": input_messages},
    config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessage):  # Filter to just model responses
        print(chunk)
        print(chunk.content, end="|")
        print()