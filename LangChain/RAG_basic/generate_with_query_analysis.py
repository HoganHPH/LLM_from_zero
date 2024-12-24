import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


from loadDocument import load_document
link = "https://en.wikipedia.org/wiki/History_of_the_United_States"
docs = load_document(link)

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

### [QUERY ANAYSIS]
### Add some metadata to the documents 
### (add some (contrived) sections to the document which are filtered on later)
total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"
# print(all_splits[0].metadata)


###
### Init vector store
###
from getEmbeddingModel import get_embedding_model
# Load embedding model from hugging face
checkpoint = "sentence-transformers/all-mpnet-base-v2"
embeddings = get_embedding_model(checkpoint)

### [QUERY ANAYSIS]
### Update the documents in our vector store
### Use InMemoryVectorStore in this case
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(all_splits)


### ===========================================================================================
### STEP 1: Load a chat model
### I use a pretrained checkpoint from HuggingFace
### ===========================================================================================
from getChatModel import get_chat_model
chat_model = get_chat_model()

### ===========================================================================================
### STEP 3: Retreive related documents and Generate answer
### I use a pretrained checkpoint from HuggingFace
### ===========================================================================================


##
## STEP 3.1: Create a prompt for RAG
##

## Custom prompt
from langchain_core.prompts import PromptTemplate

template = """
        <|system|>
        You are a helpful assistant.<|end|>
        <|user|>
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
        Context: {context}
        Question: {question}
        Helpful Answer:
        <|end|>
        <|assistant|>
"""
prompt = PromptTemplate.from_template(template)
print("\nINIT PROMPT TEMPLATE SUCCESSFFULLY!\n")

### [QUERY ANAYSIS]
### STEP 3.2: Define a schema for our search query
###

from typing import Literal, TypedDict
from typing_extensions import Annotated

class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]


###
### STEP 3.3: Init LangGraph
###

""" Note*:
    There are 3 things need to be defined:
    - The state of our application
    - The nodes of our application (i.e., application steps)
    - The "control flow" of our application (e.g., the ordering of the steps)
"""

from langchain_core.documents import Document
from typing_extensions import List, TypedDict

# Define State (controls what data is input to the application/prompt)
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

def analyze_query(state: State):
    structured_llm = chat_model.bind_tools([Search])
    query = structured_llm.invoke(state["question"]).dict()['tool_calls'][0]['args']
    return {"query": query}    


def retrieve(state: State):
    print("\n======================")
    print(state["query"])
    print("\n======================")
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = chat_model.invoke(messages)
    return {"answer": response.content}

# Define Control flow
from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()


###
### STEP 4: Invoke message and Show results
###
for step in graph.stream(
    # Beginning section
    # {"question": "When did the United States declare its independence?"},
    
    # End section
    {"question": "What does the end of the document say about Biden finished the withdrawal of American troops?"},
    stream_mode="updates",
):
    print(f"{step}\n\n----------------\n")