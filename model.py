from openai import OpenAI
import time

class LM:
    def __init__(self, api_key=None):
        if api_key is None:
            with open("open_router_api_key.txt") as f:
                api_key = f.read().strip()
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    
    def run_model(self, prompt, max_retries=3, wait_seconds=5):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="google/gemma-3n-e2b-it:free",
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(wait_seconds)
                else:
                    raise
