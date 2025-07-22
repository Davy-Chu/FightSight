import openai
import os
from dotenv import load_dotenv
import json
import hashlib
load_dotenv()
CACHE_FILE = "data/cache/llm_fall_classifications.json"
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        llm_cache = json.load(f)
else:
    llm_cache = {}

def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(llm_cache, f)

def classify_fall_event_with_llm(summary_text):
    prompt = f"""
You are an expert MMA fight analyst. Classify the type of fall described below as either:
- knockdown
- takedown
- slip

Description:
\"\"\"
{summary_text}
\"\"\"

Respond only with the classification.
"""

    hash_key = hash_text(summary_text)
    if hash_key in llm_cache:
        return llm_cache[hash_key]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You are an expert MMA fight event classifier. You are given a description of a fall event and you need to classify it as either a knockdown, takedown, or slip."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        result = response.choices[0].message.content
        llm_cache[hash_key] = result
        save_cache()
        return result
    except Exception as e:
        print(f"[ERROR] LLM classification failed: {e}")
        return "unknown"
