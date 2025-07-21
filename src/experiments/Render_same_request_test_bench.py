import csv
import openai
import concurrent.futures
import time
import os
from datetime import datetime
from src.experiments.models.enums import BenchmarkColumns, ContentType, LlmModel, graphic_card

NUMBER_OF_CLIENTS = 5
Number_of_prompts = 10
max_tokens = 10
CONTENT_TYPE = ContentType.RENDER
graphic_type = graphic_card.RTX5000
BASE_DIR = r"C:\Users\ASUS\Desktop\Pars\HardwareAware\src\experiments\data\SGLang_mohammadhusein"

CSV_FILENAME = f"{NUMBER_OF_CLIENTS}_{Number_of_prompts}_{max_tokens}_{CONTENT_TYPE.value.lower()}_{graphic_type.value}.csv"
CSV_PATH = os.path.join(BASE_DIR, CSV_FILENAME)

model = LlmModel.QWEN_15B
tokenizer = model.get_tokenizer()

client = openai.OpenAI(
    base_url="http://192.168.70.154:8080/v1",
    api_key="sk-no-key-required"
)

COLUMNS = BenchmarkColumns.get_all_columns()


def init_csv_file():
    directory = os.path.dirname(CSV_PATH)
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_exists = os.path.exists(CSV_PATH)

    with open(CSV_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or os.stat(CSV_PATH).st_size == 0:
            headers = [col[0] for col in COLUMNS]
            writer.writerow(headers)


def save_to_csv(metrics):
    row_data = []
    for col_name, col_type in COLUMNS:
        value = metrics.get(col_name)

        if col_type == "datetime" and value is not None:
            if isinstance(value, datetime):
                value = value.strftime('%H:%M:%S.%f')[:-3]  # Format as HH:MM:SS.000
            elif isinstance(value, (int, float)):
                value = datetime.fromtimestamp(value).strftime('%H:%M:%S.%f')[:-3]

        row_data.append(value)

    with open(CSV_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row_data)


def send_request(session_id, prompt):
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

    try:
        metrics['BT'] = time.time()
        stream = client.chat.completions.create(
            model=LlmModel.QWEN_15B.value,
            messages=[{"role": "system", "content": "give me 100word essay"}],
            stream=True,
            max_tokens=max_tokens,
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

            if metrics['FT'] and metrics['BT']:
                metrics['TTFT'] = (metrics['FT'] - metrics['BT']) * 1000
            if metrics['LT'] and metrics['FT']:
                metrics['TGT'] = abs((metrics['LT'] - metrics['FT']) * 1000)
            if metrics.get('TTFT') and metrics['prompt_tokens']:
                metrics['PP'] = metrics['prompt_tokens'] / ((metrics['TTFT']) / 1000)
            if metrics.get('TGT') and metrics['completion_tokens']:
                metrics['TG'] = metrics['completion_tokens'] / (metrics['TGT'] / 1000)

        save_to_csv(metrics)
        return metrics
    except Exception as e:
        metrics['error'] = str(e)
        save_to_csv(metrics)
        return metrics


def run_concurrent_sessions():
    init_csv_file()

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUMBER_OF_CLIENTS) as executor:
        futures = {executor.submit(send_request, i, "test prompt"): i
                   for i in range(NUMBER_OF_CLIENTS)}

        for future in concurrent.futures.as_completed(futures):
            session_id = futures[future]
            try:
                metrics = future.result()
                print(f"Session {session_id} completed successfully")
            except Exception as e:
                print(f"Session {session_id} failed with error: {str(e)}")


if __name__ == "__main__":
    print(f"Starting {NUMBER_OF_CLIENTS} concurrent client sessions...")
    run_concurrent_sessions()
    print(f"\nResults saved to: {CSV_PATH}")