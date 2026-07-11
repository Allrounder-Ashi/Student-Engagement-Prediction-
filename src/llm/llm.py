import os
from dotenv import load_dotenv
from google import genai
from .prompt import create_prompt

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")
client = genai.Client(api_key=API_KEY)
def generate_response(engagement, boredom, confusion, frustration, topic):
    prompt = create_prompt(
        engagement,
        boredom,
        confusion,
        frustration,
        topic
    )

    response = client.models.generate_content(
        model="models/gemini-3.5-flash",
        contents=prompt
    )

    return response.text    