import openai
from src.experiments.models.enums import QuestionsEnum

client = openai.OpenAI(
    base_url="http://192.168.70.137:8080",  # ashkan
    api_key="sk-no-key-required"
)

correct_answers = [
    "blue",
    "2",
    "2",
    "no",
    "7",
    "blue",
    "4",
    "Paris",
    "7",
    "milk"
]

correct_count = 0


def evaluate_question(question, correct_answer):
    global correct_count
    response = ""

    stream = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # ashkan
        messages=[{"role": "system", "content": question}],
        stream=True,
        max_tokens=200,
        stream_options={"include_usage": True}
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
        else:
            print(f"Metadata: {chunk.model_dump()}")

    print(response)

    if response.lower() == correct_answer.lower():
        correct_count += 1


for i, question_enum in enumerate(QuestionsEnum):
    print(f"\nQuestion {i + 1}: {question_enum.value}")
    evaluate_question(question_enum.value, correct_answers[i])
