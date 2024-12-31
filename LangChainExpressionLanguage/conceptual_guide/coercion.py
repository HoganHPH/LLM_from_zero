### =====================================================================================
### =====================================================================================
### Conceptual guide : LangChain Expression Language (LCEL)
### =====================================================================================
### =====================================================================================


###
### Coercion
###

""" !Important:

    - You have to be careful because the Mapping Dictionary is not a 
    RunnableParallel object, it is just a dictionary. This means that the 
    following code will raise an AttributeError
    
    - You have to be careful because the Lambda Function is not a RunnableLambda object, 
    it is just a function. This means that the following code will raise an AttributeError
"""

from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

def mul_two(x: int) -> int:
    return x * 2

def mul_three(x: int) -> int:
    return x * 3

runnable_1 = RunnableLambda(add_one)
runnable_2 = RunnableLambda(mul_two)
runnable_3 = RunnableLambda(mul_three)

mapping = {
    "key1": runnable_2,
    "key2": runnable_3,
}
# chain = runnable_1 | mapping 
# Equivalently:
from langchain_core.runnables import RunnableParallel, RunnableSequence
chain = RunnableSequence(runnable_1, RunnableParallel(mapping))
print(chain.invoke(20))