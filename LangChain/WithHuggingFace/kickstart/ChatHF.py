import os
from dotenv import load_dotenv

load_dotenv()

"""
Docs: 
- Link: https://python.langchain.com/docs/integrations/chat/huggingface/

Note:
- 2 ways to instantiate a ChatHuggingFace model:
    + HuggingFaceEndpoint
    + HuggingFacePipeline
    
"""

###
# HuggingFaceEndpoint
###
"""
- Note*: HuggingFaceEndpoint can used model directly without downloading
"""
# """
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)

chat_model = ChatHuggingFace(llm=llm)
print(chat_model)
# """


###
# HuggingFacePipeline
###
"""
- Note*: HuggingFacePipeline download and save cache of the model
"""
"""
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "top_k": 50,
        "temperature": 0.1,
    },
)

chat_model = ChatHuggingFace(llm=llm)
print("\n====> Non-quantized model : \n", chat_model)
"""

###
# Instatiating with Quantization
###
"""
from transformers import BitsAndBytesConfig
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)


llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
    model_kwargs={"quantization_config": quantization_config},
)

chat_model = ChatHuggingFace(llm=llm)
print("\n====> Quantized model : ", chat_model)
"""

###
# Invocation
###

"""
- Note*: There are 2 ways to invoke a message
"""

# 1st way: use directly llm model
ai_msg = llm.invoke("Football is")
print(ai_msg)


# 2nd way: create messages
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="Football is"
    ),
]

ai_msg = chat_model.invoke(messages)
print(ai_msg.content)



