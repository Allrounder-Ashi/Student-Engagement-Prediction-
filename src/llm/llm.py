import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from .prompt import create_prompt

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

client = genai.Client(api_key=API_KEY)


# 1. Define Pydantic Schema for Structured Gemini Output
class TutorResponse(BaseModel):
    explanation: str = Field(
        description="Clear, simple topic explanation tailored to student's state."
    )
    worked_example: str = Field(
        description="Exactly one concrete step-by-step worked example."
    )
    practice_question: str = Field(
        description="Exactly one practice question calibrated to student's state."
    )


# 2. Refactored Response Generation Function
def generate_response(
    engagement: str,
    boredom: str,
    confusion: str,
    frustration: str,
    topic: str
) -> TutorResponse:
    """
    Generates structured, behavior-aware tutoring responses using Gemini API.
    """
    prompt = create_prompt(
        engagement,
        boredom,
        confusion,
        frustration,
        topic
    )

    for attempt in range(2):
        try:
            # Calling Gemini with strict Pydantic response schema
            response = client.models.generate_content(
                model="gemini-2.5-flash",  # Recommended stable flash model
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TutorResponse,
                    max_output_tokens=500,
                    temperature=0.7,
                    seed=42
                )
            )

            # response.parsed is automatically converted to a TutorResponse instance
            if response.parsed:
                print("\nGEMINI RESPONSE PARSED SUCCESSFULLY:")
                print(f"Explanation: {response.parsed.explanation[:60]}...")
                return response.parsed

        except Exception as e:
            print(f"Gemini attempt {attempt + 1} failed: {e}")
            if attempt < 1:
                time.sleep(5)

    # Graceful Fallback Object if API fails
    print("[LLM Error]: Returning default fallback tutor response.")
    return TutorResponse(
        explanation=f"Let's break down {topic} into simple, easy-to-understand steps.",
        worked_example="Here is a step-by-step example to help visualize the core concept.",
        practice_question="Can you try explaining the main idea in your own words?"
    )
