import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
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
### STEP 2: Use model to invoke a message
### 2.1. Give chatbot a new information
###
response = chat_model.invoke([HumanMessage(content="Hi! My name is Hoang")])
print(f"{response.content}\n")


###
### STEP 2: 
### 2.2. Does chatbot remember the previous information?
###
response = chat_model.invoke([HumanMessage(content="What is my name?")])
print(f"{response.content}\n")


### ==> Conclusion: Basic Chatbot fails in remebering conversation history
### In Basic Chatbot, need to pass all messages (conversation history) at the same time

response = chat_model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)
print(f"{response.content}\n")