'''
Script to demonstrate Output Parsers with Gemini model using LangChain.
Using a StrOutputParser to extract specific information from the model's response.
Using library : langchain_core.output_parsers.StrOutputParser
'''
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# Define Template 1
template1 = PromptTemplate(template="Write a detailed report on {topic}", input_variables=["topic"])

# Define Template 2
template2 = PromptTemplate(template="Write a 4 point summary on the following {text}", input_variables=["text"])

# Create StrOutputParser instance
parser = StrOutputParser()

# Create a chain of operations
chain = template1 | model | parser | template2 | model | parser

response = chain.invoke({"topic":"English Premier League"})
print("Final Response from Gemini model after Output Parsing:")
print("Final Response : ", response)
print()
print("#"*100)
