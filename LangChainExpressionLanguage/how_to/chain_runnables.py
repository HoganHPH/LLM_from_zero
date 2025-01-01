### =====================================================================================
### =====================================================================================
### HOW TO : CHAIN RUNNABLES
### =====================================================================================
### =====================================================================================


###
### "|" operator
###

### Define Chat model
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)
chat_model = ChatHuggingFace(llm=llm)

### Chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

chain = prompt | chat_model | StrOutputParser()

# response = chain.invoke({"topic": "bears"})
# print(response)

###
### Coercion
###

analysis_prompt = ChatPromptTemplate.from_template("is this a funny joke? {joke}")

composed_chain = {"joke": chain} | analysis_prompt | chat_model | StrOutputParser()

# response = composed_chain.invoke({"topic": "bears"})
# print(response)

###
### Add custom logic to chains
###
composed_chain_with_lambda = (
    chain
    | (lambda input: {"joke": input})
    | analysis_prompt
    | chat_model
    | StrOutputParser()
)

# response = composed_chain_with_lambda.invoke({"topic": "beets"})
# print(response)

###
### .pipe() method
###
from langchain_core.runnables import RunnableParallel

composed_chain_with_pipe = (
    RunnableParallel({"joke": chain})
    .pipe(analysis_prompt)
    .pipe(chat_model)
    .pipe(StrOutputParser())
)

# response = composed_chain_with_pipe.invoke({"topic": "dogs"})
# print(response)

###
### Or equivalently:
###
composed_chain_with_pipe = RunnableParallel({"joke": chain}).pipe(
    analysis_prompt, chat_model, StrOutputParser()
)
response = composed_chain_with_pipe.invoke({"topic": "cats"})
print(response)