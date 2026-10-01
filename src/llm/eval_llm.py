import json
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from src.llm.llm import generate_response

# Pydantic Schema for Evaluator Output
class EvaluationResult(BaseModel):
    pedagogical_alignment: float = Field(description="Score from 1.0 to 5.0")
    concept_accuracy: float = Field(description="Score from 1.0 to 5.0")
    tone_appropriateness: float = Field(description="Score from 1.0 to 5.0")
    constraint_adherence: float = Field(description="Score from 1.0 to 5.0")
    overall_score: float = Field(description="Average score across all dimensions")
    justification: str = Field(description="Reasoning for assigned scores")

EVALUATOR_SYSTEM_PROMPT = """
You are an expert Pedagogical Evaluator for Adaptive AI Tutoring Systems.
Your job is to evaluate an AI Tutor's output generated for a specific student affective state and topic.

Rate the output on a scale of 1.0 to 5.0 across these metrics:
1. Pedagogical Alignment: Does the explanation adapt correctly (micro-steps for Confusion, real-world examples for Boredom, reassurance for Frustration, challenge for Engagement)?
2. Concept Accuracy: Is the topic explanation clear, accurate, and easy to follow?
3. Tone Appropriateness: Is the tone supportive without explicitly disclosing predicted emotional states?
4. Constraint Adherence: Is it concise, properly structured, and under 250 words?
"""

def evaluate_tutor_output(topic: str, state: dict, tutor_response) -> EvaluationResult:
    client = genai.Client()

    eval_input = f"""
    [STUDENT AFFECTIVE STATE]
    Engagement: {state['engagement']}, Boredom: {state['boredom']}, 
    Confusion: {state['confusion']}, Frustration: {state['frustration']}

    [TOPIC]
    {topic}

    [GENERATED TUTOR RESPONSE]
    Explanation: {tutor_response.explanation}
    Worked Example: {tutor_response.worked_example}
    Practice Question: {tutor_response.practice_question}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=eval_input,
        config=types.GenerateContentConfig(
            system_instruction=EVALUATOR_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=EvaluationResult,
            temperature=0.1
        )
    )
    return response.parsed

if __name__ == "__main__":
    test_cases = [
        {"topic": "Binary Search Trees", "state": {"engagement": "Low", "boredom": "Low", "confusion": "High", "frustration": "Low"}},
        {"topic": "Merge Sort Complexity", "state": {"engagement": "Low", "boredom": "High", "confusion": "Low", "frustration": "Low"}},
        {"topic": "Recursion & Call Stack", "state": {"engagement": "Low", "boredom": "Low", "confusion": "High", "frustration": "High"}},
        {"topic": "Dynamic Programming", "state": {"engagement": "High", "boredom": "Low", "confusion": "Low", "frustration": "Low"}},
    ]

    print("--- Running Automated LLM-as-a-Judge Evaluation Benchmark ---")
    for idx, tc in enumerate(test_cases, 1):
        # 1. Generate tutor output
        tutor_out = generate_response(
            tc['state']['engagement'], tc['state']['boredom'],
            tc['state']['confusion'], tc['state']['frustration'], tc['topic']
        )
        # 2. Evaluate output
        eval_res = evaluate_tutor_output(tc['topic'], tc['state'], tutor_out)
        
        print(f"\n[Test Case {idx}] Topic: {tc['topic']} | State: {tc['state']}")
        print(f"-> Pedagogical Alignment: {eval_res.pedagogical_alignment}/5.0")
        print(f"-> Concept Accuracy:      {eval_res.concept_accuracy}/5.0")
        print(f"-> Tone Appropriateness: {eval_res.tone_appropriateness}/5.0")
        print(f"-> Constraint Adherence:  {eval_res.constraint_adherence}/5.0")
        print(f"-> Overall Score:        {eval_res.overall_score}/5.0")
        print(f"-> Reason: {eval_res.justification}")
