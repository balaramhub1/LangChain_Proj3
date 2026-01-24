'''
Script to demonstrate interacting with llm like a chatbot using LangChain.
'''

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

chat_history = [
    SystemMessage(content="You are a helpful assistant.")
    ]

while True:
    user_input = input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting chat.")
        break
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI: ", response.content)

print("Chat History : ", chat_history)
