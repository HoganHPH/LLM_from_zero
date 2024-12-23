import os
import subprocess

from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("HF_TOKEN")

subprocess.run(["huggingface-cli", "login", "--token", TOKEN])
print("LOGIN HUGGINGFACE SUCCESSFULLY!")

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
print("SET UP LANGSMITH DONE!")

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
print("SET UP TAVILY DONE!")