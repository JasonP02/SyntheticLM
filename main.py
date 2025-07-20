# Load model directly
import json
import argparse
import openai
import random
import re

# Set up OpenRouter API
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = "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"  # <-- Put your OpenRouter API key here

# At top of file - easy to modify during experiments
class Config:
    def __init__(self):
        self.num_seed_tasks = 8
        self.human_seed_ratio = 0.8
        self.max_iterations = 1
        self.model_name = "google/gemma-3n-e2b-it:free"
        self.max_tokens = 25
        
cfg = Config()

def load_seed_tasks(path="seed_tasks.jsonl"):
    """Load seed tasks from JSONL file"""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

human_seed_tasks = load_seed_tasks("seed_tasks.jsonl")

class PoolFilter:
    def __init__(self):
        pass        
    
    def filter_classifications(self):
        pass
    
    def filter_regression(self):
        pass

class LM:
    def __init__(self, api_key=None):
        self.client = openai.OpenAI(
            api_key=api_key or openai.api_key,  # Use parameter or global
            base_url="https://openrouter.ai/api/v1"
        )

    def build_prompt(self, prompt_type, **kwargs):
        """Centralized prompt builder"""
        templates = {
            'new_task': self._new_task_template,
            'classification': self._classification_template,
            'instance': self._instance_template
        }
        return templates[prompt_type](**kwargs)

    def run_model(self, prompt):
        response = self.client.chat.completions.create(
            model=cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        reply = response.choices[0].message.content
        return reply

    def new_instruction_prompt(self, task_type:str, pool_text:str) -> str:
        """
        Simple prompt generator depending on the step in self-instruct pipeline
        Args:
            human_made_task: str
            task_type: str
        Returns:
            str: prompt for the model
        """
        if task_type == "new_task":
            return f"""Come up with a series of tasks that are unique. There should be both classification and non-classification tasks. Follow after Task {cfg.num_seed_tasks}. Follow the existing format with no changes or commentary.\n{pool_text}"""
        elif task_type == "instruction_classification":
            # TODO: seperate pool text based on "Task:"
            return f"""Can the following task be regarded as a classification task with finite output labels? For each task, return a yes/no answer.\n{pool_text}\n"""
        elif task_type == "input_first_instance_generation":
            # TODO:
            return f"""Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly. \n {pool_text}\n"""
        elif task_type == "output_first_instance_generation":
            return f"""Given the classification task definition and the class labels, generate an input that corresponds to each of the class labels. If the task doesn't require input, \njust generate the correct class label.\n {pool_text}"""
        else:
            raise ValueError(f"Invalid task type: {task_type}")
        
    def create_classification_prompt(self, prompt_dict):
        tasks = prompt_dict["instruction"]
        classes = prompt_dict["is_classification"]
        
        prompt = "Can the following task be regarded as a classification task with finite output labels? Follow the provided format with no elaboration. Only return classification for Tasks that have a number.\n"
        
        for i in range(len(tasks)):
            label = "Yes" if classes[i] else "No"
            prompt += f"Task: {tasks[i]}\nIs it classification? {label}\n"
        
        return prompt
    
def parse_task_class(classification_input_prompt, output_classes):
    # Extract tasks from classification_input_prompt
    task_lines = re.findall(r"Task:\s*(.+)", classification_input_prompt)
    
    # Extract class labels from output_classes (last word in each line)
    class_lines = [line.strip() for line in output_classes.split('\n') if line.strip().lower().startswith("classification:")]
    class_labels = [line.split()[-1] for line in class_lines]  # "Yes" or "No"
    
    # Pair them up
    return [{"task": task, "class": label} for task, label in zip(task_lines, class_labels)]
            

def sample_seeds_weighted(human_tasks, llm_tasks, total_needed, human_ratio=0.8):
    """Sample seed tasks with ratio of human and llm generated tasks
    Args:
        human_tasks: list; a list of human generated tasks
        llm_tasks: list; a list of llm generated tasks
        total_required: int; number of output pool samples for pipeline
        human_ratio: float; percentage of pool tasks which are human made
        
    Returns: Indices of human_tasks and llm_taks to be extracted from the source data"""
    
    num_human = int(total_needed * human_ratio)
    num_llm = total_needed - num_human
    
    # Sample from human seed tasks
    if human_tasks and num_human > 0:
        human_sample = random.sample(human_tasks, min(num_human, len(human_tasks)))
    
    # Sample from LLM tasks
    if llm_tasks and num_llm > 0:
        llm_sample = random.sample(llm_tasks, min(num_llm, len(llm_tasks)))
    
    return human_sample, llm_sample

# loop: sample from task pool, generate tasks, classify, model response (in/out dependent), filter, add to pool, repeat
llm_seed_tasks = []

for i in range(cfg.max_iterations):
    model = LM()
    
    # Determine ratio based on iteration
    if len(llm_seed_tasks) == 0:
        ratio = 1.0  # 100% human for bootstrap
    else:
        ratio = cfg.human_seed_ratio  # 80% human for subsequent
    
    human_sample_indices, llm_sample_indices = sample_seeds_weighted(
        human_seed_tasks, 
        llm_seed_tasks, 
        cfg.num_seed_tasks, 
        ratio
    )


    # 1. Sample from task pool
    if len(llm_seed_tasks) == 0:
        # bootstrapping
        prompt_samples_as_list = [human_seed_tasks[human_sample_indices[i]] for i in range(cfg.num_seed_tasks)]
        prompt_samples = {}
        for key in prompt_samples_as_list[0].keys():
            prompt_samples[key] = [sample[key] for sample in prompt_samples_as_list]
        
        # initiate model
        # parse text and add new lines
        pool_text = ""
        for idx, task in enumerate(prompt_samples["instruction"], 1):
            pool_text += f"Task {idx}: {task}\n"
            
        # Initiate step 1
        pool_input_prompt = model.new_instruction_prompt("new_task", pool_text)

        # Obtain pool prompts from LM for classification
        output_tasks = model.run_model(pool_input_prompt)


        # Step 2 : Classification
        # Task: 
        # Is it classification?
        # We need to pass in the prompt samples, and corresponding class - then construct the prompt accordingly

        
        classification_input_prompt = model.create_classification_prompt(prompt_samples)
        output_classes = model.run_model(classification_input_prompt + output_tasks)

        # Step 3: Generate data based on the task type
        # First, we need to split up the output_classes based on whether they are classification or not, then feed into model with distinct prompts

        parsed_class_data = parse_task_class(classification_input_prompt, output_classes)
        print(parsed_class_data)

        # parsed_class_data is a list of dicts: [{"task": ..., "class": ...}, ...]

        classification_tasks = [item["task"] for item in parsed_class_data if item["class"].lower().startswith("y")]
        regression_tasks = [item["task"] for item in parsed_class_data if item["class"].lower().startswith("n")]

        print("Classification tasks:", classification_tasks)
        print("Regression tasks:", regression_tasks)
        
    else: # We use a 80/20 split of human seed tasks and LLM seed tasks for new generations
        pass