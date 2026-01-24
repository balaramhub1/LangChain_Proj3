'''
Script to demonstrate Structured output parsing with Gemini model using LangChain.
Using a Pydantic model to define output schema and parse the model's response.
'''

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers.structured import ResponseSchema, OutputFixingParser
from langchain_core.output_parsers import StructuredOutputParser
from langchain_core.output_parsers.fix import OutputFixingParser

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

schema = [
    ResponseSchema(name="fact1", description="first fact about blackhole"),
    ResponseSchema(name="fact2", description="second fact about blackhole"),
    ResponseSchema(name="fact3", description="third fact about blackhole")]

parser = StructuredOutputParser.from_response_schemas(schema)

safe_parser = OutputFixingParser.from_llm(model, parser)

template = PromptTemplate(template='''
give me 3 facts about {topic}. Return only valid json instructions whihc follows this fromat\n{format_instructions}''',
                          input_variables=["topic"],
                          partial_variables={"response_format": parser.get_format_instructions()}
                          )

chain = template | safe_parser | model

response = chain.invoke({"topic":"blackhole"})
print("Final Structured Response from Gemini model after Output Parsing:")
print("Final Response : ", response)
print()
print("#"*100)
