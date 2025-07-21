import random
import openai
import concurrent.futures
import time
import os
import csv
from datetime import datetime
from src.experiments.models.enums import BenchmarkColumns, ContentType, graphic_card, LlmModel, SendingType, \
     EngineType, PromptMode

CLIENT_COUNTS = [1, 10, 30, 50]
random.seed(42)
PROMPT_LENGTHS = [256, 512, 1024]
CONTENT_TYPE = ContentType.RENDER
graphic_type = graphic_card.RTX5000
MAX_TOKENS = 2
BASE_DIR = r"C:\Users\ASUS\Desktop\Pars\HardwareAware\src\experiments\data\Llama_cpp_Ashkan_final"

client = openai.OpenAI(
    base_url="http://192.168.70.137:8080",
    #base_url="http://192.168.70.154:8080/v1",
    api_key="sk-no-key-required"
)
model = LlmModel.QWEN_15B
tokenizer = model.get_tokenizer()
sending_type = SendingType.CONCURRENT
prompt_mode = PromptMode.SAME
engine_name = EngineType.LLAMA
COLUMNS = [col[0] for col in BenchmarkColumns.get_all_columns()]


def init_csv_file(client_count, prompt_length):
    csv_filename = f"{client_count}_{prompt_length}_{MAX_TOKENS}_{graphic_type.value}_{sending_type.value}_{prompt_mode.value}_{CONTENT_TYPE.value.lower()}_{engine_name.value.lower()}.csv"
    csv_path = os.path.join(BASE_DIR, csv_filename)

    directory = os.path.dirname(csv_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
            writer.writeheader()

    return csv_path


def save_to_csv(metrics, csv_path):
    saved_metrics = {k: v for k, v in metrics.items() if k in COLUMNS}

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
        writer.writerow(saved_metrics)


def generate_prompt(prompt_length):
    vocab = list(tokenizer.get_vocab().keys())
    selected_tokens = random.sample(vocab, prompt_length)
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(selected_tokens))


def send_request(session_id, csv_path, prompt):
    metrics = {
        'session_id': session_id,
        'BT': time.time(),
        'FT': None,
        'LT': None,
        'TGT': None,
        'PP': None,
        'TG': None,
        'TTFT': None,
        'completion_tokens': None,
        'prompt_tokens': None,
        'prompt_ms': None,
        'prompt_per_token_ms': None,
        'prompt_per_second': None,
        'predicted_ms': None,
        'predicted_per_token_ms': None,
        'predicted_per_second': None,
        'content': "",
        'error': None
    }

    # try:
    metrics['BT'] = time.time()
    stream = client.chat.completions.create(
        model=LlmModel.QWEN_15B.value,
        messages=[{"role": "system", "content": prompt}],
        stream=True,
        max_tokens=MAX_TOKENS,
        temperature=0.8,
        top_p=0.95,
        stream_options={"include_usage": True}
    )

    for chunk in stream:
        print("#", chunk.model_dump())
        if metrics['FT'] is None and chunk.choices[0].delta.content:
            metrics['FT'] = time.time()

        if len(chunk.choices) and chunk.choices[0].delta.content:
            metrics['content'] += chunk.choices[0].delta.content

        print(hasattr(chunk, 'usage'))
        if hasattr(chunk, 'usage'):
            print(chunk.model_dump())
            usage_data = chunk.usage
            timings_data = chunk.model_extra.get('timings', {}) if hasattr(chunk, 'model_extra') else {}

            metrics.update({
                'prompt_tokens': usage_data.prompt_tokens if hasattr(usage_data, 'prompt_tokens') else None,
                'completion_tokens': usage_data.completion_tokens if hasattr(usage_data,
                                                                             'completion_tokens') else None,
                'prompt_ms': timings_data.get('prompt_ms'),
                'prompt_per_token_ms': timings_data.get('prompt_per_token_ms'),
                'prompt_per_second': timings_data.get('prompt_per_second'),
                'predicted_ms': timings_data.get('predicted_ms'),
                'predicted_per_token_ms': timings_data.get('predicted_per_token_ms'),
                'predicted_per_second': timings_data.get('predicted_per_second')
            })

        if metrics['LT'] is None and chunk.choices[0].finish_reason:
            metrics['LT'] = time.time()
            # break

        if metrics['FT'] and metrics['BT']:
            metrics['TTFT'] = (metrics['FT'] - metrics['BT']) * 1000
        if metrics['LT'] and metrics['FT']:
            metrics['TGT'] = abs((metrics['LT'] - metrics['FT']) * 1000)
        if metrics.get('TTFT') and metrics['prompt_tokens']:
            metrics['PP'] = metrics['prompt_tokens'] / ((metrics['TTFT']) / 1000)
        if metrics.get('TGT') and metrics['completion_tokens']:
            metrics['TG'] = metrics['completion_tokens'] / (metrics['TGT'] / 1000)

    for time_field in ['BT', 'FT', 'LT']:
        if metrics[time_field] is not None:
            metrics[time_field] = datetime.fromtimestamp(metrics[time_field])

    save_to_csv(metrics, csv_path)
    return metrics


def run_benchmark_for_config(client_count, prompt_length):
    print(f"\nStarting benchmark with {client_count} clients and {prompt_length} token prompt...")
    prompt = []
    for i in range(client_count):
        prompt.append(generate_prompt(prompt_length))
    csv_path = init_csv_file(client_count, prompt_length)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=client_count) as executor:
            index_multiplyer = 0
            if prompt_mode.value == "VARIANT":
                index_multiplyer = 1

            futures = {}
            if sending_type.value == "D":
                print("sleeeeep...")
                for i in range(client_count):
                    time.sleep(0.05)
                    futures[executor.submit(send_request, i, csv_path, prompt[i * index_multiplyer])] = i
            else:
                futures = {executor.submit(send_request, i, csv_path, prompt[i * index_multiplyer]): i
                       for i in range(client_count)}
            for future in concurrent.futures.as_completed(futures):

                session_id = futures[future]
                future.result()
                print(f"Session {session_id} completed successfully")
    finally:
        print(f"Results saved to: {csv_path}")


def run_all_benchmarks():
    print("Starting comprehensive benchmark tests...")

    for client_count in CLIENT_COUNTS:
        for prompt_length in PROMPT_LENGTHS:
            time.sleep(3)
            print(
                f'next Benchmark {client_count}_{prompt_length}_{MAX_TOKENS}_{CONTENT_TYPE.value.lower()} will start 3s later')
            run_benchmark_for_config(client_count, prompt_length)

    print("\nAll benchmark tests completed!")


if __name__ == "__main__":
    run_all_benchmarks()
