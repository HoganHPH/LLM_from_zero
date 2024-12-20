import getpass
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dotenv import load_dotenv

 
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

###
# STEP 1: Documents Loader
###

#
# Example 1: Using 'Document' to define documents
# 
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]


#
# Example 2: Load PDF file
# 

from langchain_community.document_loaders import PyPDFLoader

file_path = "History_of_the_United_States.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
# print(len(docs))
# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)


###
# STEP 2: Splitting
###

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
# print("Length of splits : ", len(all_splits))
# print(all_splits[0])

###
# STEP 3: Embeddings
###

from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model from hugging face
checkpoint = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=checkpoint)

# Embed text to vector
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
# print(f"Generated vectors of length {len(vector_1)}\n")
# print(vector_1[:10])

###
# STEP 4: Vector stores
###

from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import urllib 
# ca = certifi.where()

# Connect to MongoDB

## Config
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME")
MONGODB_PASSWORD = urllib.parse.quote_plus(os.getenv("MONGODB_PASSWORD"))

MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
MONGODB_ATLAS_CLUSTER_URI = MONGODB_ATLAS_CLUSTER_URI.format(MONGODB_USERNAME, MONGODB_PASSWORD)
# print(MONGODB_ATLAS_CLUSTER_URI)

## initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
# print("\n\n", client)

print("CONNECT TO MONGODB SUCCESSFULLY!")

MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")
MONGODB_VECTOR_SEARCH_INDEX_NAME = os.getenv("MONGODB_VECTOR_SEARCH_INDEX_NAME")

MONGODB_COLLECTION = client[MONGODB_DB_NAME][MONGODB_COLLECTION_NAME]

vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embeddings,
    index_name=MONGODB_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)

print("CREATE VECTOR STORE SUCCESSFULLY!")

# index the documents.
ids = vector_store.add_documents(documents=all_splits)

## Usage

question = "When was the Province of Carolina established?"

# 1st way:
# results = vector_store.similarity_search(question)
# print("\n\n", results[0])

# 2nd way - Return scores:
# results = vector_store.similarity_search_with_score(question)
# doc, score = results[0]
# print(f"\nScore: {score}\n")
# print(doc)


###
# STEP 5: Retrievers
###


# 1st way - Using chain
# from typing import List

# from langchain.globals import set_debug
# set_debug(True)
# from langchain_core.documents import Document
# from langchain_core.runnables import chain


# @chain
# def retriever(query: str) -> List[Document]:
#     return vector_store.similarity_search(query, k=1)


# question1 = "When was the Province of Carolina established?"
# question2 = "When was New Netherland established?"

# outputs = retriever.batch(
#     [
#         question1,
#         question2,
#     ],
# )
# print(outputs)


# 2nd: Using vector_store
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

outputs = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
print(outputs)