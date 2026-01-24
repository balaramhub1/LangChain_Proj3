'''
Script to demonstrate Structured Output with Gemini model using LangChain.
Using TypedDict from typing module to define output schema.
'''

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# output schema definition
class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

prompt = '''
This hardware is great, but the software is terrible, so many boiler plate apps. and my phone keeps hanging when i play PUBG.'''

response = structured_model.invoke(prompt)
print("Structured Response from Gemini model:")
print("Response : ", response)
print()
print("Response without strcutured output:")
raw_response = model.invoke(prompt)
print("Raw Response : ", raw_response.content)
print()

