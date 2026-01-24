'''
Script to demonstrate
'''
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# Chat Template with MessagePlaceHolder
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", '{query}')
])

# Load chat history from txt file (chatbot_history.txt)

chat_history=[]
with open("../../resources/chatbot_history.txt", "r") as file:
    lines = file.readlines()
    chat_history.extend(lines)

print("Chat History Loaded from File: ", chat_history)

prompt = chat_template.invoke(
    {
        "chat_history": chat_history,
        "query": "where is my refund?"
    }
)

print("Chat Prompt with History : ",prompt)



