# Load model directly
import json
import argparse
import openai
import random
import re
from dataclasses import dataclass

# Set up OpenRouter API
openai.api_base = "https://openrouter.ai/api/v1"
with open("openrouter_api_key.txt") as f:
    openai.api_key = f.read().strip()  # <-- Loads your key securely

class LM:
    def __init__(self, api_key=None):
        self.client = openai.OpenAI(
            api_key=api_key or openai.api_key,  # Use parameter or global
            base_url="https://openrouter.ai/api/v1"
        )
        self.cfg = Config()


    def run_model(self, prompt):
        response = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        reply = response.choices[0].message.content
        return reply

@dataclass
class Config:
    num_seed_tasks: int = 8
    human_seed_ratio: float = 0.8
    max_iterations: int = 1
    model_name: str = "google/gemma-3n-e2b-it:free"
    max_tokens: int = 25


class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()
    
    def load_data(self):
        """Load data from a JSONL file"""
        with open(self.data_path, 'r') as f:
            return [json.loads(line) for line in f if line.strip()]
    
    def save_data_as_jsonl(self, data, path=None):
        """Save data to a JSONL file"""
        path = path or self.data_path
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    def parse_task_class(self, classification_input_prompt, output_classes):
        # Extract tasks from classification_input_prompt
        task_lines = re.findall(r"Task:\s*(.+)", classification_input_prompt)
        
        # Extract class labels from output_classes (last word in each line)
        class_lines = [line.strip() for line in output_classes.split('\n') if line.strip().lower().startswith("classification:")]
        class_labels = [line.split()[-1] for line in class_lines]  # "Yes" or "No"
        
        # Pair them up
        return [{"task": task, "class": label} for task, label in zip(task_lines, class_labels)]


class PoolFilter:
    def __init__(self):
        pass        
    
    def filter_classifications(self):
        pass
    
    def filter_regression(self):
        pass

    
class PromptCreator:
    def __init__(self):
        pass

    def create_pool_task_prompt(self, pool_text, config):
        """
        Create a prompt for the model to generate new tasks based on the pool text.
        """
        return f"""Come up with a series of tasks that are unique. There should be both classification and non-classification tasks. Follow after Task {config.num_seed_tasks}. Follow the existing format with no changes or commentary.\n{pool_text}"""

    def create_classification_prompt(self, prompt_dict):
        """
        Create a prompt for the model to classify tasks as classification or not.
        """
        tasks = prompt_dict["instruction"]
        classes = prompt_dict["is_classification"]
        
        prompt = "Can the following task be regarded as a classification task with finite output labels? Follow the provided format with no elaboration. Only return classification for Tasks that have a number.\n"
        
        for i in range(len(tasks)):
            label = "Yes" if classes[i] else "No"
            prompt += f"Task: {tasks[i]}\nIs it classification? {label}\n"
        
        return prompt

    def create_instance_generation_prompt(self, prompt_dict, task_type):
        """
        Create a prompt for the model to generate instances based on task type.
        """
        tasks = prompt_dict["instruction"]
        
        if task_type == "input_first":
            return f"""Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly. \n {tasks}\n"""
        elif task_type == "output_first":
            return f"""Given the classification task definition and the class labels, generate an input that corresponds to each of the class labels. If the task doesn't require input, just generate the correct class label.\n {tasks}"""
        else:
            raise ValueError(f"Invalid task type: {task_type}")

class Pipeline:
    def __init__(self):
        self.data_processor = DataProcessor("seed_tasks.jsonl")
        self.prompt_creator = PromptCreator()
        self.pool_filter = PoolFilter()
        self.llm_seed_tasks = []
        self.human_seed_tasks = self.data_processor.load_data()
        self.human_seed_tasks = DataProcessor("seed_tasks.jsonl").data
        self.model = LM(api_key=openai.api_key)
        self.cfg = Config()

    def get_pool_tasks(self, human_tasks, llm_tasks, total_needed):
        """Sample seed tasks with ratio of human and llm generated tasks"""

        if len(llm_tasks) < 10:
            human_ratio = 1.0
        else:
            human_ratio = 0.8

        num_human = int(total_needed * human_ratio)
        num_llm = total_needed - num_human

        human_sample_indices = random.sample(human_tasks, min(num_human, len(human_tasks)))

        # Sample from LLM tasks
        if llm_tasks and num_llm > 0:
            llm_sample_indices = random.sample(llm_tasks, min(num_llm, len(llm_tasks)))
        else:
            llm_sample_indices = []
            
        human_pool_prompts = [task["instruction"] for task in human_sample_indices]
        llm_pool_prompts = [task["instruction"] for task in llm_sample_indices]

        return {
            "instruction": human_pool_prompts + llm_pool_prompts,
            "is_classification": [task["is_classification"] for task in human_sample_indices] + [task["is_classification"] for task in llm_sample_indices]
        }
            
    def format_pool_prompts(self, prompt_samples):
        """
        Obtain pool prompts from the model for classification.
        """
        pool_text = ""
        for idx, task in enumerate(prompt_samples["instruction"], 1):
            pool_text += f"Task {idx}: {task}\n"

        pool_input_prompt = self.prompt_creator.create_pool_task_prompt(pool_text, self.cfg)
        return pool_input_prompt
    
    def run_pipeline(self):
        for i in range(self.cfg.max_iterations):

            sample_tasks = self.get_pool_tasks(
                self.human_seed_tasks,
                self.llm_seed_tasks,
                self.cfg.num_seed_tasks,
            )

            formatted_pool_tasks = self.format_pool_prompts(sample_tasks)
            print("Formatted Pool Tasks:", formatted_pool_tasks)
            output_tasks = self.model.run_model(formatted_pool_tasks)

            # Step 2 : Classification
            # Task: 
            # Is it classification?
            # We need to pass in the prompt samples, and corresponding class - then construct the prompt accordingly


            classification_input_prompt = self.prompt_creator.create_classification_prompt(output_tasks)
            output_classes = self.model.run_model(classification_input_prompt + output_tasks)

            # Step 3: Generate data based on the task type
            # First, we need to split up the output_classes based on whether they are classification or not, then feed into model with distinct prompts

            parsed_class_data = self.data_processor.parse_task_class(classification_input_prompt, output_classes)
            print(parsed_class_data)

            # parsed_class_data is a list of dicts: [{"task": ..., "class": ...}, ...]

            classification_tasks = [item["task"] for item in parsed_class_data if item["class"].lower().startswith("y")]
            regression_tasks = [item["task"] for item in parsed_class_data if item["class"].lower().startswith("n")]

            print("Classification tasks:", classification_tasks)
            print("Regression tasks:", regression_tasks)

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run_pipeline()
    print("Pipeline completed successfully.")