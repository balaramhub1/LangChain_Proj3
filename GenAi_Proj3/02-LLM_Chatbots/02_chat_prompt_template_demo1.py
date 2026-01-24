'''
Script to demonstrate
using prompt templates with chat models in LangChain.
'''

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain the concept of {concept} in simple terms.")
])

prompt = chat_template.invoke(
    {
        "domain": "artificial intelligence",
        "concept": "machine learning"
    }
)

print("Chat Prompt : ",prompt)

response = model.invoke(prompt)
print("Response from Gemini model using prompt template:")
print(response.content)
print()
