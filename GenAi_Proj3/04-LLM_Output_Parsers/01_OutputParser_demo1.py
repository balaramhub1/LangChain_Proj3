'''
Script to demonstrate Output Parsers with Gemini model using LangChain.
Using a custom output parser to extract specific information from the model's response.
'''

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# Define Template 1
template1 = PromptTemplate(template="Write a detailed report on {topic}", input_variables=["topic"])
# Define prompt1 - using template1
prompt1 = template1.invoke({"topic":"English Premier League"})
# response1 from model
response1 = model.invoke(prompt1).content
print("Response 1 from Gemini model:")
print("Response : ", response1)
print()
print("#"*100)


# Define Template 2
template2 = PromptTemplate(template="Write a 4 point summary on the following {text}", input_variables=["text"])
# Define prompt2 - using template2
prompt2 = template2.invoke({"text":str(response1)})
# response2 from model
response2 = model.invoke(prompt2).content
print("Response 2 from Gemini model:")
print("Response : ", response2)
print()
print("#"*100)
