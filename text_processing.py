import json
import re
from dataclasses import dataclass

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
        class_lines = [line.strip() for line in output_classes.split('\n') if line.strip()]
        class_labels = [line.split()[-1] for line in class_lines]  # "Yes" or "No"

        print(f"Classes found: {class_labels}")
        print(f"Tasks found: {task_lines}")

        # Pair them up
        return [{"task": task, "class": label} for task, label in zip(task_lines, class_labels)]

class PromptCreator:
    def __init__(self):
        pass

    def create_pool_task_prompt(self, pool_text, config):
        """
        Create a prompt for the model to generate new tasks based on the pool text.
        """
        return f"""Come up with a series of tasks that are unique. There should be both classification and non-classification tasks. The task shall have enough information to return an answer. Follow after Task {config.num_seed_tasks}. Follow the existing format with no changes or commentary.\n{pool_text}"""

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
        classification = prompt_dict["class"]

        
        if classification == "No":
            return f"""Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly. \n {tasks}\n"""
        elif classification == "Yes":
            return f"""Given the classification task definition and the class labels, generate an input that corresponds to each of the class labels. If the task doesn't require input, just generate the correct class label.\n {tasks}"""
        else:
            raise ValueError(f"Invalid task type: {task_type}")

class PoolFilter:
    def __init__(self):
        pass        
    
    def filter_classifications(self):
        pass
    
    def filter_regression(self):
        pass