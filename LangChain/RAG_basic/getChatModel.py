import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


def get_chat_model():
    llm = HuggingFaceEndpoint(
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        task="text-generation",
        max_new_tokens=100,
        do_sample=False,
    )
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model