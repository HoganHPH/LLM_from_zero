###
### Step 1: Define LangChain chat model from a LLM with checkpoint of Hugging Face
###
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)

chat_model = ChatHuggingFace(llm=llm)
# print(chat_model)

""" 
    **Note**:
    
    Specifically, we can define:
        - Possible values for each property
        - Description to make sure that the model understands the property
        - Required properties to be returned
"""

###
### Step 2: Define chat promt (looks like an instruction for LLM)
###

from langchain_core.prompts import ChatPromptTemplate


tagging_prompt = ChatPromptTemplate.from_template(
    """
        <|system|>
        You are a helpful assistant.<|end|>
        <|user|>
        Extract the desired information from the following passage.
        Only extract the properties mentioned in the 'Classification' function.
        Passage:
        {input}
        <|end|>
        <|assistant|>
    """
)


###
### Step 3: Define model (properties LLM need to return)
###
from pydantic import BaseModel, Field
class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")
   

###
### Step 4: Assign/Bind task for LLM
### 
tagging_llm = chat_model.bind_tools([Classification])
# print(tagging_llm)

###
### Step 5: Function calling
### 
inp_neg = "I am very angry because it is rainy today!"
inp_pos = "I am very happy because it is sunny today!"
prompt = tagging_prompt.invoke({"input": inp_neg})
# print(prompt)

response = tagging_llm.invoke(prompt).dict()
print(response['tool_calls'])

