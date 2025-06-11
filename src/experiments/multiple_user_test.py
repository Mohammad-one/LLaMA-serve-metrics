import openai
import concurrent.futures
import time

client = openai.OpenAI(
    base_url="http://192.168.70.137:8080",
    api_key="sk-no-key-required"
)

# Create a list of different messages for each session
"""messages_list = [
    {"role": "system", "content": "Give me a 100-word essay about the capital of italy."},
    {"role": "system", "content": "Give me a 100-word essay about the Eiffel Tower."},
    {"role": "system", "content": "Describe spain in 100 words."},
    {"role": "system", "content": "What is the history of tehran in 100 words?"},
    {"role": "system", "content": "Provide a brief history of the usa in 100 words."},
    {"role": "system", "content": "Give a 100-word essay on norvey culture."},
    {"role": "system", "content": "Explain the significance of algeria in 100 words."},
    {"role": "system", "content": "Describe the geography of london in 100 words."},
    {"role": "system", "content": "Give a brief introduction to jordan cuisine in 100 words."},
    {"role": "system", "content": "Explain the role of Paris in greece history in 100 words."}
]"""

messages_list = [
    {"role": "system", "content": "Provide a detailed analysis of the political, economic, and cultural influence of Italy's capital, Rome, on European and global affairs over the past century."},
    {"role": "system", "content": "Describe the architectural evolution of the Eiffel Tower, including its original design, modifications over time, and its impact on modern architecture."},
    {"role": "system", "content": "Give a c omprehensive history of Spain, focusing on its role in the global exploration and colonial expansion during the Age of Discovery, as well as its impact on modern European politics."},
    {"role": "system", "content": "Discuss the history of Tehran, including its evolution from a small town to the capital of Iran, its cultural significance, and the major political events that have shaped its modern identity."},
    {"role": "system", "content": "Provide a detailed overview of the USA’s history, focusing on the founding principles, the Civil War, the emergence as a global superpower, and its impact on contemporary international relations."},
    {"role": "system", "content": "Analyze the cultural heritage of Norway, including its Viking origins, the evolution of Scandinavian culture, and its influence on modern European art, architecture, and politics."},
    {"role": "system", "content": "Explain the significance of Algeria's role in the African continent, covering its colonial history, the war of independence, and its contemporary geopolitical influence in the Mediterranean and Sub-Saharan Africa."},
    {"role": "system", "content": "Describe the geography of London, examining its historical development along the Thames, its role as a global financial hub, and the impact of its urban planning and public transportation systems on other world cities."},
    {"role": "system", "content": "Give a comprehensive introduction to Jordanian cuisine, including its origins, the influence of Bedouin culture, and its integration of diverse Middle Eastern culinary traditions over time."},
    {"role": "system", "content": "Analyze the role of Paris in shaping the history of Greece, focusing on the intellectual and cultural exchanges between the two cities during the 19th and 20th centuries, and the contributions of Parisian philosophers to Greek thought."}
]

def send_request(session_id, message):
    start_time = time.time()
    try:
        completion = client.chat.completions.create(
            model="facebook/opt-350m",
            messages=[message]  # Use the specific message for this session
        )
        response_text = completion.choices[0].message.content
        end_time = time.time()
        return session_id, 200, response_text, end_time - start_time
    except Exception as e:
        end_time = time.time()
        return session_id, 500, str(e), end_time - start_time


def run_sessions():
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(send_request, i, messages_list[i - 1]): i for i in range(1, 11)}
        for future in concurrent.futures.as_completed(futures):
            session_id, status_code, response, elapsed_time = future.result()
            print(f"[Session {session_id}] Status: {status_code}")
            #if you want to print the response, please uncomment the below line
            #print(f"[Session {session_id}] Response: {response}")
            print(f"[Session {session_id}] End Time: {elapsed_time:.2f} seconds\n")

if __name__ == "__main__":
    start = time.time()
    run_sessions()
    print(f"\nAll requests completed in {time.time() - start:.2f} seconds")

