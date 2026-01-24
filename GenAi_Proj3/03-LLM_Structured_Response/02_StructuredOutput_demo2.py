'''
Script to demonstrate Structured Output with Gemini model using LangChain.
Library used TypedDict and Annotated from typing module to define output schema with descriptions.
'''

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import TypedDict, Annotated, Optional

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# output schema definition
class Review(TypedDict):
    key_themes: Annotated[list[str], "List of key themes mentioned in the review"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "Overall sentiment of the review: positive, negative, or neutral"]
    pros: Annotated[Optional[list[str]], "List of pros mentioned in the review"]
    cons: Annotated[Optional[list[str]], "List of cons mentioned in the review"]

prompt = '''
For context I had all the Pixel Flagships from the Pixel 1 - 4 but primarily
used them a few years after release. I also have a Pixel 7 so I can talk
about the recent experience.

Generally the Pixels in my opinion are (for better or worse) very close to
the "classic" iOS experience. Very stable, very smooth - "shit just works".
However that comes at the cost of out the box customization options, they are
pretty weak, weaker even than recent iOS Versions and nowhere close to what
for example Samsung offers.

Another thing to keep in mind is that most Google Apps like Google Photos,
Drive, Maps, ... work just as well on other phones. The perks you used to get
with earlier Pixels like Unlimited Google Photos backups or access to Google
One features are sadly long gone with newer models (though they still work on
the Pixel 1 for example).

Also worth mentioning is that theoretical and gaming performance of recent
models lacks far behind the competition. If you are planning to do lots of
heavy 3D gaming, Pixel phones in general are best to be avoided. Even a
iPhone 13 probably has the Pixel 9 Pro beat in terms of gaming performance.

That being said, the old Pixel phones I have like the 1, 2XL, 3, 4 ... all
have aged extremely well when it comes to day to day speed compared to other
phones with the same chipset, so in a way in regular use the Pixel generally
punches above its weight and never randomly bogs down like older Samsung or
Xiaomi devices (although they also have gotten better over the years).
'''

structured_model = model.with_structured_output(Review)
response = structured_model.invoke(prompt)
print("Structured Response from Gemini model:")
print("Response : ", response)
print()
print("#"*100)
print("Response without strcutured output:")
raw_response = model.invoke(prompt)
print("Raw Response : ", raw_response.content)
print()