### =====================================================================================
### =====================================================================================
### Conceptual guide : LangChain Expression Language (LCEL)
### =====================================================================================
### =====================================================================================


###
### Composition Primitives: RunnableSequence and RunnableParallel
###

### =====================================================================================
### RunnableSequence
print("\nRunnable Sequence\n\n")

from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

def mul_two(x: int) -> int:
    return x * 2

runnable_1 = RunnableLambda(add_one)
runnable_2 = RunnableLambda(mul_two)

## Without chain:
output1 = runnable_1.invoke(10)
final_output = runnable_2.invoke(output1)
print(final_output)

## With chain:
chain = runnable_1 | runnable_2
# Or equivalently:
# from langchain_core.runnables import RunnableSequence
# sequence = RunnableSequence(first=runnable_1, last=runnable_2)

print(chain.invoke(1)) # (1 + 1) * 2

# Or batch
print(chain.batch([1, 2, 3]))

### =====================================================================================
### RunnableParallel
print("\nRunnable Parallel\n\n")

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

chain = runnable_1 | {  # this dict is coerced to a RunnableParallel
    "mul_two": runnable_2,
    "mul_three": runnable_3,
}
# Or equivalently:
# sequence = runnable_1 | RunnableParallel(
#     {"mul_two": runnable_2, "mul_three": runnable_3}
# )
# Also equivalently:
# sequence = runnable_1 | RunnableParallel(
#     mul_two=runnable_2,
#     mul_three=runnable_3,
# )

print(chain.invoke(1)) # (1+1)*2 and (1+1)*3 at the same time => Return 2 results

print(chain.batch([1, 2, 3])) # Return 2 results for each batch sample