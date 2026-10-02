import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from .prompt import create_prompt


# 1. Resolve absolute path directly to root .env file
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT_DIR / ".env"

# 2. Force load_dotenv to read your .env file directly
load_dotenv(dotenv_path=ENV_PATH, override=True)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(f"GEMINI_API_KEY not found. Attempted loading from: {ENV_PATH}")

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


def generate_response_with_status(
    engagement, boredom, confusion, frustration, topic
):
    prompt = create_prompt(
        engagement, boredom, confusion, frustration, topic
    )

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TutorResponse,
                    max_output_tokens=500,
                    temperature=0.7,
                    seed=42,
                ),
            )

            if response.parsed is not None:
                return response.parsed, False

        except Exception as e:
            print(f"Gemini attempt {attempt + 1} failed: {e}")
            if attempt < 1:
                time.sleep(5)

    fallback = TutorResponse(
        explanation=f"Let's break down {topic} into simple, easy-to-understand steps.",
        worked_example="Here is a step-by-step example to help visualize the core concept.",
        practice_question="Can you try explaining the main idea in your own words?",
    )
    return fallback, True


def generate_response(
    engagement, boredom, confusion, frustration, topic
):
    response, is_fallback = generate_response_with_status(
        engagement, boredom, confusion, frustration, topic
    )
    return response
