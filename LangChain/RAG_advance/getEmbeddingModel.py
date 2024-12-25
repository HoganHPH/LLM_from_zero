from langchain_huggingface import HuggingFaceEmbeddings


# Load embedding model from hugging face
def get_embedding_model(checkpoint="sentence-transformers/all-mpnet-base-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=checkpoint)
    print("\nLOAD EMBEDDING MODEL SUCCESSFULLY!\n")
    return embeddings