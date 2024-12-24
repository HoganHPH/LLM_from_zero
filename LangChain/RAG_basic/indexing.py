import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


### ===========================================================================================
### STEP 2: INDEXING
### This step is an abbreviated version of the content in the "semantic search" section
### ===========================================================================================


###
### STEP 2.1: Load document
### Document: https://en.wikipedia.org/wiki/History_of_the_United_States
###
from loadDocument import load_document
link = "https://en.wikipedia.org/wiki/History_of_the_United_States"
docs = load_document(link)

# print(docs[0].page_content[:500])


###
### STEP 2.2: Split documents into chunks
###
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"\nSplit blog post into {len(all_splits)} sub-documents.\n")


###
### STEP 2.3: Embed chunks
### Init embedding mode that embed the contents of each document split into embedding vectors
###
from getEmbeddingModel import get_embedding_model

# Load embedding model from hugging face
checkpoint = "sentence-transformers/all-mpnet-base-v2"
embeddings = get_embedding_model(checkpoint)


###
### STEP 2.4: Define vector store
### I use MongoDB as a vector store to store embedding vectors
###

from getVectorStore import get_mongodb_vector_store

vector_store = get_mongodb_vector_store(embeddings)


###
### STEP 2.5: Store the embeddings in vector store
###
""" Note*:
    Run only the first time to store the vectors
"""

document_ids = vector_store.add_documents(documents=all_splits)
print(document_ids[:3])
print("\nSTORE EMBEDDING VECTORS SUCCESSFFULLY!\n")