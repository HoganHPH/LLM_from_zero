# LangChain Agents

import os
from dotenv import load_dotenv


# Init Tavily search engine API
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


###
### STEP 1: Define tools
### main tool of choice will be Tavily - a search engine
###

from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)

# Test API
# search_results = search.invoke("what is the weather in SF")
# print(search_results)
""" Note*:
    If we want, we can create other tools.
    Once we have all the tools we want, we can put them in a list that we will reference later.
"""
tools = [search]


###
### STEP 2: Define a LLM ChatModel using HuggingFace checkpoint
### Here I use a HunggingFace checkpoint to create model
###

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)
chat_model = ChatHuggingFace(llm=llm)

# Test chat model
# response = chat_model.invoke([HumanMessage(content="hi!")])
# print(response.content)


###
### STEP 3: Enable this model to do tool calling
### use .bind_tools to give the language model knowledge of these tools
###
chat_model_with_tools = chat_model.bind_tools(tools)

# Test chat model with tools using normal message
# response = chat_model_with_tools.invoke([HumanMessage(content="Hi!")])
# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}\n")

# Test chat model with tools using message that would expect a tool to be called.
# response = chat_model_with_tools.invoke([HumanMessage(content="What's the weather in SF?")])
# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}\n")


""" Note*:
    This isn't calling that tool yet - it's just telling us to. 
    In order to actually call it, we'll want to create our agent.
"""


###
### STEP 4: Create the Agent
### using LangGraph to construct the agent.
###

### 4.1. Initialize the agent
""" Note*:
    Passing in the model, not model_with_tools
    because create_react_agent will call .bind_tools for us under the hood
"""
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(chat_model, tools)

### 4.2. Run the agent
""" Note*:
    Note that for now, these are all stateless queries 
    (it won't remember previous interactions). 
    Note that the agent will return the final state at the end of the interaction 
    (which includes any inputs, we will see later on how to get only the outputs).
"""

# Test when there's no need to call a tool:
# response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})
# print(response["messages"])

# response = agent_executor.invoke(
#     {"messages": [HumanMessage(content="whats the weather in sf?")]}
# )
# for msg in response["messages"]:
#     print(msg)
#     print()


###
### STEP 5: Streaming messages
### If the agent executes multiple steps, this may take a while. 
### To show intermediate progress, we can stream back messages as they occur.
###

# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content="whats the weather in sf?")]}
# ):
#     print(chunk)
#     print("----")


###
### STEP 6: Streaming tokens
###
""" Note*
    There is a bug in calling async await
"""

###
### STEP 7: Adding in memory
### As mentioned earlier, this agent is stateless. 
### This means it does not remember previous interactions. 
### To give it memory we need to pass in a checkpointer.
###

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent_executor = create_react_agent(chat_model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}


for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob!")]}, config
):
    print(chunk)
    print("----")
    

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats my name?")]}, config
):
    print(chunk)
    print("----")