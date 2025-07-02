import openai

client = openai.OpenAI(
    base_url="http://192.168.88.130:8080",
    api_key="sk-no-key-required"
)

stream = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    messages=[
        {"role": "system", "content": "give me a 200 word essay"}
    ],
    stream=True,
    max_tokens=200,
    stream_options={
        "include_usage": True
    }
)


i = 0
for chunk in stream:
    i += 1
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    else:
        print()
        print(chunk.model_dump())

# import time
#
# import openai
#
# client = openai.OpenAI(
#     base_url="http://192.168.88.130:8080",
#     api_key="sk-no-key-required"
# )
#
# start_time = time.time()
#
# try:
#     completion = client.chat.completions.create(
#         model="facebook/opt-350m",
#         messages=[
#             {"role": "system", "content": "What is the history of Tehran in 100 words?"}
#         ]
#     )
#     # Ensure response is valid
#     if completion.choices:
#         print(completion.choices[0].message.content)
#     else:
#         print("No response received from the model.")
#
# except Exception as e:
#     print(f"An error occurred: {e}")
#
# end_time = time.time()
# print(f"Response processed in {end_time - start_time:.2f} seconds")
