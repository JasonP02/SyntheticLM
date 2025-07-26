from openai import OpenAI
from pipeline import Pipeline

# Set up OpenRouter API
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=open("open_router_api_key.txt").read().strip()
)

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run_pipeline()
    print("Pipeline completed successfully.") 
