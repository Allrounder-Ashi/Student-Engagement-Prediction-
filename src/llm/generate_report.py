from .llm import generate_response
def generate_learning_report(
    engagement,
    boredom,
    confusion,
    frustration,
    topic
):
    report = generate_response(
        engagement,
        boredom,
        confusion,
        frustration,
        topic
    )

    return report