import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, AIMessage

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
### STEP 2: Define chatbot app using memory workflow of LangGraph
###

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

# Define the function that calls the model
def call_model(state: MessagesState):
    response = chat_model.invoke(state["messages"])
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)


# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


###
### STEP 3: Define configuration to identify the speaker (Who is speaking?)
### Note*: have to define and pass into every time 
###
config = {"configurable": {"thread_id": "user_1"}}


###
### STEP 4: Use defined chatbot app to invoke human message 
###

# 4.1.
query = "Hi! My name is Hoang."
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

# 4.2.
query = "What's my name?"
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()