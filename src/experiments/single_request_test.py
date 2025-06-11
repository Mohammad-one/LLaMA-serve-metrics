"""import openai

client = openai.OpenAI(
    base_url="http://192.168.70.137:8080",
    api_key = "sk-no-key-required"
)

completion = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "system", "content": "What is the history of tehran in 100 words?"}
    ]
)
print(completion.choices[0].message.content)
"""
import openai
import time

client = openai.OpenAI(
    base_url="http://192.168.88.130:8080",
    api_key="sk-no-key-required"
)

start_time = time.time()

try:
    # Make the API request
    completion = client.chat.completions.create(
        model="facebook/opt-350m",
        messages=[
            {"role": "system", "content": "What is the history of Tehran in 100 words?"}
        ]
    )

    # Ensure response is valid
    if completion.choices:
        print(completion.choices[0].message.content)
    else:
        print("No response received from the model.")

except Exception as e:
    print(f"An error occurred: {e}")

end_time = time.time()
print(f"Response processed in {end_time - start_time:.2f} seconds")
