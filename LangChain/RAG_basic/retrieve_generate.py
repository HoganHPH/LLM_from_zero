import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


###
### Init vector store
###
from getEmbeddingModel import get_embedding_model
# Load embedding model from hugging face
checkpoint = "sentence-transformers/all-mpnet-base-v2"
embeddings = get_embedding_model(checkpoint)

from getVectorStore import get_mongodb_vector_store
vector_store = get_mongodb_vector_store(embeddings)

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


###
### STEP 3.1: Create a prompt for RAG
###

# from langchain import hub
# prompt = hub.pull("rlm/rag-prompt")


### Or custom prompt
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

""" Example:
    example_messages = prompt.invoke(
        {"context": "(context goes here)", "question": "(question goes here)"}
    ).to_messages()
    print(example_messages[0].content)
"""


###
### STEP 3.2: Init LangGraph
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
    context: List[Document]
    answer: str
    
# Define Nodes (application steps)
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    print(messages)
    response = chat_model.invoke(messages)
    return {"answer": response.content}

# Define Control flow
from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


###
### STEP 3.3: Run
###

result = graph.invoke({"question": "When did the United States declare its independence?"})
# result = graph.invoke({"question": "Who lived from 5,500 BCE to 600 CE?"})
# result = graph.invoke({"question": "Who lived from 100000 BCE to 80000 BCE?"})
# result = graph.invoke({"question": "What is my name"}) # Case AI don't know
# print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')


###
### Addition: Stream mode
###
"""
for step in graph.stream(
    {"question": "Who lived from 5,500 BCE to 600 CE?"}, stream_mode="updates"
):
    print(f"{step}\n\n----------------\n")
"""