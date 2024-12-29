### ===================================================================================
### ===================================================================================
###
### How to pass run time values to tools?
###
### ===================================================================================
### ===================================================================================

"""
Docs:
    - Prevent the model from generating certain tool arguments and 
    injecting them in directly at runtime.
    - The LLM should only control the parameters of the tool 
    that are meant to be controlled by the LLM, while other parameters 
    (such as user ID) should be fixed by the application logic
"""

import os
from dotenv import load_dotenv
load_dotenv()


###
### Define Chat model
###

### OpenAI
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

### MistralAI
# from langchain_mistralai import ChatMistralAI
# llm = ChatMistralAI(model="mistral-large-latest")


###
### Hiding arguments from the model
###

from typing import List

from langchain_core.tools import InjectedToolArg, tool
from typing_extensions import Annotated

user_to_pets = {}


@tool(parse_docstring=True)
def update_favorite_pets(
    pets: List[str], 
    user_id: Annotated[str, InjectedToolArg]
) -> None:
    """Add the list of favorite pets.

    Args:
        pets: List of favorite pets to set.
        user_id: User's ID.
    """
    user_to_pets[user_id] = pets

@tool(parse_docstring=True)
def delete_favorite_pets(user_id: Annotated[str, InjectedToolArg]) -> None:
    """Delete the list of favorite pets.

    Args:
        user_id: User's ID.
    """
    if user_id in user_to_pets:
        del user_to_pets[user_id]
        
@tool(parse_docstring=True)
def list_favorite_pets(user_id: Annotated[str, InjectedToolArg]) -> None:
    """List favorite pets if any.

    Args:
        user_id: User's ID.
    """
    return user_to_pets.get(user_id, [])


# print(update_favorite_pets.get_input_schema().schema())
# print(update_favorite_pets.tool_call_schema.schema())

""" !Note
    - If we look at the input schemas for these tools, we'll see that user_id is still listed
    - But if we look at the tool call schema, which is what is passed to the model for tool-calling, 
    user_id has been removed
"""

### When we invoke our tool, we need to pass in user_id:
user_id = "123"
update_favorite_pets.invoke({"pets": ["lizard", "dog"], "user_id": user_id})
# print(user_to_pets)
# print(list_favorite_pets.invoke({"user_id": user_id}))

### But when the model calls the tool, no user_id argument will be generated
tools = [
    update_favorite_pets,
    delete_favorite_pets,
    list_favorite_pets,
]
# llm_with_tools = llm.bind_tools(tools)
# ai_msg = llm_with_tools.invoke("my favorite animals are cats and parrots")
# print(ai_msg.tool_calls)
# print(tool_calls)

###
### Injecting arguments at runtime
###

### If we want to actually execute our tools using the model-generated tool call, 
### we'll need to inject the user_id ourselves
from copy import deepcopy
from langchain_core.runnables import chain

@chain
def inject_user_id(ai_msg):
    tool_calls = []
    for tool_call in ai_msg.tool_calls:
        tool_call_copy = deepcopy(tool_call)
        tool_call_copy["args"]["user_id"] = user_id
        tool_calls.append(tool_call_copy)
    return tool_calls


# print(inject_user_id.invoke(ai_msg))

### now we can chain together our model, injection code, 
### and the actual tools to create a tool-executing chain

tool_map = {tool.name: tool for tool in tools}

@chain
def tool_router(tool_call):
    return tool_map[tool_call["name"]]

chain = llm_with_tools | inject_user_id | tool_router.map()
# response = chain.invoke("my favorite animals are cats and dogs")
# print(response)
# print(user_to_pets)

###
### Other ways of annotating args
###

### There are 3 ways ways of annotating our tool args:

### 1st way:
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class UpdateFavoritePetsSchema(BaseModel):
    """Update list of favorite pets"""

    pets: List[str] = Field(..., description="List of favorite pets to set.")
    user_id: Annotated[str, InjectedToolArg] = Field(..., description="User's ID.")

@tool(args_schema=UpdateFavoritePetsSchema)
def update_favorite_pets(pets, user_id):
    user_to_pets[user_id] = pets

# print(update_favorite_pets.get_input_schema().schema())
# print(update_favorite_pets.tool_call_schema.schema())

### 2nd way:
from typing import Optional, Type


class UpdateFavoritePets(BaseTool):
    name: str = "update_favorite_pets"
    description: str = "Update list of favorite pets"
    args_schema: Optional[Type[BaseModel]] = UpdateFavoritePetsSchema

    def _run(self, pets, user_id):
        user_to_pets[user_id] = pets


# print(UpdateFavoritePets().get_input_schema().schema())
# print(UpdateFavoritePets().tool_call_schema.schema())

### 3rd way:
class UpdateFavoritePets2(BaseTool):
    name: str = "update_favorite_pets"
    description: str = "Update list of favorite pets"

    def _run(self, pets: List[str], user_id: Annotated[str, InjectedToolArg]) -> None:
        user_to_pets[user_id] = pets

print(UpdateFavoritePets2().get_input_schema().schema())
print(UpdateFavoritePets2().tool_call_schema.schema())