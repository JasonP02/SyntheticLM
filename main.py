# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
from synthetic_data.model_prompts import new_instruction_prompt

if __name__ == "main":
    
    def run_model(prompt):
        generator = pipeline("text-generation", model="EleutherAI/pythia-70m-deduped")
        result = generator(prompt, truncation=True, max_new_tokens=100, max_length=100)
        print(result[0]['generated_text'])
        

    def create_task_pool():
        prompt = data["new_instruction"]['instruction']
        print(prompt)
        run_model(prompt)
        
    create_task_pool()