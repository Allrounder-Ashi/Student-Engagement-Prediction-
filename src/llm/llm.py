import os
import time
from dotenv import load_dotenv
from google import genai
from .prompt import create_prompt

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found")
client = genai.Client(api_key=API_KEY)


def generate_response(engagement, boredom, confusion, frustration, topic):

    prompt = create_prompt(
        engagement,
        boredom,
        confusion,
        frustration,
        topic
    )

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            print("\nGEMINI RESPONSE:")
            print(response.text)

            return response.text

        except Exception as e:
            print(f"Gemini attempt {attempt+1} failed:", e)

            if attempt < 2:
                time.sleep(20)

    return "Unable to generate learning report right now."