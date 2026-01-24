'''
Script to demonstrate llm interactions using langchain.
'''
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

response = model.invoke("Whats the capital of France?")
print(response)
print("#"*100)
print(response.content)
