
import time
from google import genai
from google.genai import types, errors
from pydantic import BaseModel, Field
from src.llm.llm import generate_response_with_status
import re


# Pydantic Schema for Evaluator Output
class EvaluationResult(BaseModel):
    pedagogical_alignment: float = Field(
        ge=1.0, le=5.0,
        description="Score from 1.0 to 5.0"
    )
    concept_accuracy: float = Field(
        ge=1.0, le=5.0,
        description="Score from 1.0 to 5.0"
    )
    tone_appropriateness: float = Field(
        ge=1.0, le=5.0,
        description="Score from 1.0 to 5.0"
    )
    constraint_adherence: float = Field(
        ge=1.0, le=5.0,
        description="Score from 1.0 to 5.0"
    )
    overall_score: float = Field(
        ge=1.0, le=5.0,
        description="Average score across all dimensions"
    )
    justification: str = Field(
        description="Reasoning for assigned scores"
    )


MODEL_NAME = "gemini-3.8-flash"
MAX_ATTEMPTS = 3

EVALUATOR_SYSTEM_PROMPT = """
You are an expert Pedagogical Evaluator for Adaptive AI Tutoring Systems.
Your job is to evaluate an AI Tutor's output generated for a
specific student affective state and topic.

Rate the output on a scale of 1.0 to 5.0 across these metrics:

1. Pedagogical Alignment:
Does the explanation adapt correctly
(micro-steps for Confusion, real-world examples for
Boredom, reassurance for Frustration, challenge for Engagement)?

2. Concept Accuracy:
Is the topic explanation clear, accurate, and easy to follow?

3. Tone Appropriateness:
Is the tone supportive without explicitly disclosing
predicted emotional states?

4. Constraint Adherence:
Is it concise, properly structured, and under 250 words?

Give an independent score for each dimension.
The overall score must be the arithmetic mean of
the four dimension scores, rounded to two decimal places.
Provide a concise justification based on the response.
"""


def evaluate_tutor_output(
    topic: str,
    state: dict,
    tutor_response,
    client
) -> EvaluationResult | None:

    eval_input = f"""
    [STUDENT AFFECTIVE STATE]
    Engagement: {state['engagement']}
    Boredom: {state['boredom']}
    Confusion: {state['confusion']}
    Frustration: {state['frustration']}

    [TOPIC]
    {topic}

    [GENERATED TUTOR RESPONSE]
    Explanation: {tutor_response.explanation}
    Worked Example: {tutor_response.worked_example}
    Practice Question: {tutor_response.practice_question}
    """

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=eval_input,
                config=types.GenerateContentConfig(
                    system_instruction=EVALUATOR_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=EvaluationResult,
                    temperature=0.1
                )
            )

            result = response.parsed

            if result is None:
                print("Gemini returned no parsed evaluation.")
                return None

            # Recalculate overall score consistently
            scores = [
                result.pedagogical_alignment,
                result.concept_accuracy,
                result.tone_appropriateness,
                result.constraint_adherence
            ]

            result.overall_score = round(sum(scores) / len(scores), 2)

            return result

        
        except errors.APIError as e:
            status = getattr(e, "code", None)

            print(f"Gemini evaluation attempt {attempt} failed: {e}")

            if status not in (429, 500, 502, 503, 504):
                print("Non-retryable API error.")
                return None

            if attempt < MAX_ATTEMPTS:
                if status == 429:
                    # Respect the retry delay reported by Gemini
                    match = re.search(
                        r"retry in (\d+(?:\.\d+)?)s",
                        str(e),
                        re.IGNORECASE
                    )

                    wait_time = (
                        float(match.group(1)) + 2
                        if match else 60
                    )
                else:
                    # Backoff for temporary server overload
                    wait_time = 5 * (2 ** (attempt - 1))

                print(
                    f"Retrying in {wait_time:.1f} seconds..."
                )
                time.sleep(wait_time)
            else:
                print("Evaluation failed after all retries.")
                return None


if __name__ == "__main__":

    test_cases = [
        {
            "topic": "Binary Search Trees",
            "state": {
                "engagement": "Low",
                "boredom": "Low",
                "confusion": "High",
                "frustration": "Low"
            }
        },
        {
            "topic": "Merge Sort Complexity",
            "state": {
                "engagement": "Low",
                "boredom": "High",
                "confusion": "Low",
                "frustration": "Low"
            }
        },
        {
            "topic": "Recursion & Call Stack",
            "state": {
                "engagement": "Low",
                "boredom": "Low",
                "confusion": "High",
                "frustration": "High"
            }
        },
        {
            "topic": "Dynamic Programming",
            "state": {
                "engagement": "High",
                "boredom": "Low",
                "confusion": "Low",
                "frustration": "Low"
            }
        }
    ]

    print(
        "--- Running Automated LLM-as-a-Judge "
        "Evaluation Benchmark ---"
    )

    client = genai.Client()

    results = []
    failed_cases = 0

    for idx, tc in enumerate(test_cases, 1):

        print(f"\n[Test Case {idx}] Topic: {tc['topic']}")
        print(f"Student State: {tc['state']}")

        try:
            
            
            # 1. Generate tutor output and check its source
            tutor_out, is_fallback = generate_response_with_status(
                tc['state']['engagement'],
                tc['state']['boredom'],
                tc['state']['confusion'],
                tc['state']['frustration'],
                tc['topic']
            )

            # 2. Skip fallback responses
            if is_fallback:
                print(
                    "Tutor generation failed. "
                    "Skipping this evaluation."
                )
                failed_cases += 1
                continue

            # 3. Evaluate only genuine Gemini tutor responses
            eval_res = evaluate_tutor_output(
                tc['topic'],
                tc['state'],
                tutor_out,
                client
            )

            # 4. Skip if evaluation fails
            if eval_res is None:
                print("Evaluation unavailable. Skipping.")
                failed_cases += 1
                continue


            # 2. Evaluate tutor output
            eval_res = evaluate_tutor_output(
                tc['topic'],
                tc['state'],
                tutor_out,
                client
            )

            if eval_res is None:
                print("Evaluation unavailable. Skipping.")
                failed_cases += 1
                continue

            # 3. Display evaluation
            print(
                f"-> Pedagogical Alignment: "
                f"{eval_res.pedagogical_alignment}/5.0"
            )
            print(
                f"-> Concept Accuracy: "
                f"{eval_res.concept_accuracy}/5.0"
            )
            print(
                f"-> Tone Appropriateness: "
                f"{eval_res.tone_appropriateness}/5.0"
            )
            print(
                f"-> Constraint Adherence: "
                f"{eval_res.constraint_adherence}/5.0"
            )
            print(
                f"-> Overall Score: "
                f"{eval_res.overall_score}/5.0"
            )
            print(f"-> Reason: {eval_res.justification}")

            results.append({
                "test_case": idx,
                "topic": tc["topic"],
                "state": tc["state"],
                "pedagogical_alignment":
                    eval_res.pedagogical_alignment,
                "concept_accuracy":
                    eval_res.concept_accuracy,
                "tone_appropriateness":
                    eval_res.tone_appropriateness,
                "constraint_adherence":
                    eval_res.constraint_adherence,
                "overall_score": eval_res.overall_score,
                "justification": eval_res.justification
            })

        except Exception as e:
            print(f"Test case {idx} failed: {e}")
            failed_cases += 1

    # 4. Benchmark summary
    print("\n--- BENCHMARK SUMMARY ---")
    print(f"Total test cases: {len(test_cases)}")
    print(f"Successfully evaluated: {len(results)}")
    print(f"Failed / skipped: {failed_cases}")

    if results:
        avg_score = sum(
            r["overall_score"] for r in results
        ) / len(results)

        print(f"Average overall score: {avg_score:.2f}/5.0")

        # Save actual evaluation results
        with open("evaluation_results.json", "w") as f:
            import json
            json.dump(results, f, indent=4)

        print("Results saved to evaluation_results.json")
    else:
        print("No successful evaluations to report.")
