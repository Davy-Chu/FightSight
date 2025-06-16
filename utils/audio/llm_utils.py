import openai
from dotenv import load_dotenv
import os

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_commentary_with_llm(transcribed_text: str) -> list:
    prompt = f"""
You are analyzing MMA fight commentary. Your goal is to detect when knockdowns or takedowns from the text. 

Return a JSON array in the following format:
[
  {{ "type": "knockdown", "timestamp": "02:14", "context": "Fighter A drops Fighter B with a right hook." }},
  {{ "type": "takedown", "timestamp": "04:39", "context": "Fighter B executes a double-leg takedown." }}
]
If no knockdowns or takedowns are detected, return a string saying "No knockdowns or takedowns detected."
Here's the transcription:
\"\"\"
{transcribed_text}
\"\"\"
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    # Extract and return parsed event list
    return response.choices[0].message.content
