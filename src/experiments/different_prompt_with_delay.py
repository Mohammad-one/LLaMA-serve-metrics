import random
import openai
import concurrent.futures
import time
import os
import csv
from datetime import datetime
from src.experiments.models.enums import BenchmarkColumns, ContentType
from transformers import AutoTokenizer

CLIENT_COUNTS = [10, 50, 100]
PROMPT_LENGTHS = [512, 1024, 2048]
CONTENT_TYPE = ContentType.RENDER
MAX_TOKENS = 1
BASE_DIR = r"C:\Users\ASUS\Desktop\Pars\HardwareAware\src\experiments\data\rendering\different_prompt_with_delay"

client = openai.OpenAI(
    base_url="http://192.168.70.137:8080",
    # base_url="http://192.168.88.33:8080/v1",
    api_key="sk-no-key-required"
)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")  # ashkan
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")#peyman

COLUMNS = [col[0] for col in BenchmarkColumns.get_all_columns()]


def init_csv_file(client_count, prompt_length):
    csv_filename = f"{client_count}_{prompt_length}_{MAX_TOKENS}_{CONTENT_TYPE.value.lower()}.csv"
    csv_path = os.path.join(BASE_DIR, csv_filename)

    directory = os.path.dirname(csv_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
            writer.writeheader()

    return csv_path


def save_to_csv(metrics, csv_path):
    saved_metrics = {k: v for k, v in metrics.items() if k in COLUMNS}

    for col_name, value in saved_metrics.items():
        if value is not None and isinstance(value, datetime):
            saved_metrics[col_name] = value.strftime('%Y-%m-%d %H:%M:%S.%f')

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
        writer.writerow(saved_metrics)


def generate_prompt(prompt_length, seed=None):
    vocab = list(tokenizer.get_vocab().keys())
    if seed is not None:
        random.seed(seed)
    selected_tokens = random.sample(vocab, prompt_length)
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(selected_tokens))
# def generate_prompt(prompt_length):
#     vocab = list(tokenizer.get_vocab().keys())
#     selected_tokens = random.sample(vocab, prompt_length)
#     return tokenizer.decode(tokenizer.convert_tokens_to_ids(selected_tokens))

def send_request(session_id, csv_path, prompt):
    metrics = {
        'session_id': session_id,
        'BT': time.time(),
        'FT': None,
        'LT': None,
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

    try:
        stream = client.chat.completions.create(
            # model="Qwen/Qwen3-0.6B", #peyman
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            messages=[{"role": "system", "content": prompt}],
            stream=True,
            max_tokens=MAX_TOKENS,
            stream_options={"include_usage": True}
        )

        for chunk in stream:
            if metrics['FT'] is None and chunk.choices[0].delta.content:
                metrics['FT'] = time.time()

            if chunk.choices[0].delta.content:
                metrics['content'] += chunk.choices[0].delta.content
            else:
                if hasattr(chunk, 'usage') and hasattr(chunk, 'model_extra'):
                    usage_data = chunk.usage
                    timings_data = chunk.model_extra.get('timings', {}) if hasattr(chunk, 'model_extra') else {}

                    metrics.update({
                        'prompt_tokens': usage_data.prompt_tokens if hasattr(usage_data, 'prompt_tokens') else None,
                        'prompt_ms': timings_data.get('prompt_ms'),
                        'prompt_per_token_ms': timings_data.get('prompt_per_token_ms'),
                        'prompt_per_second': timings_data.get('prompt_per_second'),
                        'predicted_ms': timings_data.get('predicted_ms'),
                        'predicted_per_token_ms': timings_data.get('predicted_per_token_ms'),
                        'predicted_per_second': timings_data.get('predicted_per_second')
                    })

            if chunk.choices[0].finish_reason:
                metrics['LT'] = time.time()
                break

        if metrics['FT'] and metrics['BT']:
            metrics['TTFT'] = (metrics['FT'] - metrics['BT']) * 1000
        if metrics['LT'] and metrics['FT']:
            metrics['TGT'] = (metrics['LT'] - metrics['FT']) * 1000
        if metrics.get('TTFT') and metrics['prompt_tokens']:
            metrics['PP'] = metrics['prompt_tokens'] / ((metrics['TTFT'])/1000)

    except Exception as e:
        metrics['error'] = str(e)

    for time_field in ['BT', 'FT', 'LT']:
        if metrics[time_field] is not None:
            metrics[time_field] = datetime.fromtimestamp(metrics[time_field])

    metrics['content_sample'] = (metrics['content'][:100]) if metrics['content'] else None

    save_to_csv(metrics, csv_path)
    return metrics


def run_benchmark_for_config(client_count, prompt_length):
    print(f"\nStarting benchmark with {client_count} clients and {prompt_length} token prompt...")

    csv_path = init_csv_file(client_count, prompt_length)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=client_count) as executor:
            futures = {}
            for i in range(client_count):
                # Generate unique prompt for each client using i as seed
                prompt = generate_prompt(prompt_length, seed=i)
                future = executor.submit(send_request, i, csv_path, prompt)
                futures[future] = i

            for future in concurrent.futures.as_completed(futures):
                session_id = futures[future]
                try:
                    future.result()
                    print(f"Session {session_id} completed successfully")
                except Exception as e:
                    print(f"Session {session_id} failed with error: {str(e)}")
    finally:
        print(f"Results saved to: {csv_path}")


def run_all_benchmarks():
    print("Starting comprehensive benchmark tests...")

    for client_count in CLIENT_COUNTS:
        for prompt_length in PROMPT_LENGTHS:
            run_benchmark_for_config(client_count, prompt_length)

    print("\nAll benchmark tests completed!")


if __name__ == "__main__":
    run_all_benchmarks()
