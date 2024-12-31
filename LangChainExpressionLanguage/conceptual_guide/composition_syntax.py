### =====================================================================================
### =====================================================================================
### Conceptual guide : LangChain Expression Language (LCEL)
### =====================================================================================
### =====================================================================================


###
### Composition Syntax: "|" operator and .pipe method
###

from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

def mul_two(x: int) -> int:
    return x * 2

runnable_1 = RunnableLambda(add_one)
runnable_2 = RunnableLambda(mul_two)

### =====================================================================================
### "|" operator


# chain = runnable_1 | runnable_2
# Or equivalently:
from langchain_core.runnables import RunnableSequence
chain = RunnableSequence(runnable_1, runnable_2)

print(chain.invoke(2))

### =====================================================================================
### .pipe method
chain = runnable_1.pipe(runnable_2)
print(chain.invoke(2))