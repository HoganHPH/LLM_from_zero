import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# ============================================================================

###
### Load Chat Model
###
from getChatModel import get_chat_model
chat_model = get_chat_model()

###
### Load Embedding Model
###
from getEmbeddingModel import get_embedding_model
embeddings = get_embedding_model()

###
### Load Vector Store
###
from getVectorStore import get_mongodb_vector_store
vector_store = get_mongodb_vector_store(embeddings)

print("\nSET UP DONE\n")

### ============================================================================
### CHAIN
### ============================================================================

###
### Recap: Indexing documents
###

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from loadDocument import load_document

# Load documents
docs = load_document()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks and store
# Note: just run only 1 time
# _ = vector_store.add_documents(documents=all_splits)

###
### Init a State with LangGraph
###

from langgraph.graph import MessagesState, StateGraph
graph_builder = StateGraph(MessagesState)

###
### Define a TOOL that creates new retrieval step
###

from langchain_core.tools import tool

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

""" Graph will consist of three nodes:
    - A node that fields the user input, either generating a query for the retriever or responding directly;
    - A node for the retriever tool that executes the retrieval step;
    - A node that generates the final response using the retrieved context.
"""

###
### 3 step to generate new prompt
###

""" Note*:
    - ToolNode, that executes the tool and adds the result as a ToolMessage to the state.
"""

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = chat_model.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Only use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = chat_model.invoke(prompt)
    return {"messages": [response]}

###
### Compile into a signle graph object
###

from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

###
### Run
###

# input_message = "Who lived from 5,500 BCE to 600 CE?"
# for step in graph.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()
    
###
### Stateful management of chat history
###   

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# # Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}

# input_message = "Who lived from 5,500 BCE to 600 CE?"
# for step in graph.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode="values",
#     config=config,
# ):
#     step["messages"][-1].pretty_print()


# input_message = "Can you look up another one who also lives in that time?"
# for step in graph.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode="values",
#     config=config,
# ):
#     step["messages"][-1].pretty_print()
    
    
### ============================================================================
### AGENTS
### ============================================================================

from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(chat_model, [retrieve], checkpointer=memory)


config = {"configurable": {"thread_id": "def234"}}

input_message = (
    "Who lived from 5,500 BCE to 600 CE?\n\n"
    "Once you get the answer, look up common extensions of that."
)

for event in agent_executor.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    event["messages"][-1].pretty_print()

"""
    Note that the agent:

    - Generates a query to search for a standard method for task decomposition;
    - Receiving the answer, generates a second query to search for common extensions of it;
    - Having received all necessary context, answers the question.
"""