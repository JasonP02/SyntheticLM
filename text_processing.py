from fnmatch import filter
import json
import re
import random
from dataclasses import dataclass
import pyparsing as pp
import evaluate

@dataclass
class Config:
    num_seed_tasks: int = 8
    human_seed_ratio: float = 0.8
    max_iterations: int = 1
    model_name: str = "google/gemma-3n-e2b-it:free"
    len_threshold = 150
    bad_words = ['picture', 'image', 'graph']
    rouge_threshold = 0.7


class JsonUtils:
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

class PromptBases:
    def __init__(self):
        pass

    def create_pool_task_prompt(self, pool_text, config):
        """
        Create a prompt for the model to generate new tasks based on the pool text.
        """
        return f"""Come up with a series of tasks that are unique. There should be both classification and non-classification tasks; do not label them. The task shall have enough information to return an answer. Follow after Task {config.num_seed_tasks}. Follow the existing format with no changes or commentary.\n{pool_text}"""

    def create_classification_prompt(self, prompt_dict):
        """
        Create a prompt for the model to classify tasks as classification or not.
        """
        tasks = prompt_dict["instruction"]
        classes = prompt_dict["is_classification"]

        prompt = "Can the following task be regarded as a classification task with finite output labels? Follow the existing format.\n"

        for i in range(len(tasks)):
            label = "Yes" if classes[i] else "No"
            prompt += f"Task: {tasks[i]}\nIs it classification? {label}\n"

        return prompt

    def create_instance_generation_prompt(self, regression_tasks, classification_tasks):
        """
        Create a prompt for the model to generate instances based on task type.
        """
        regression_prompt = f"Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly. Provide all information. For example, do not do (put poem text here), but instead insert th poem. \n {regression_tasks}\n"
        classification_prompt = f"Given the classification task definition and the class labels, generate an input that corresponds to each of the class labels. If the task doesn't require input, just generate the correct class label.\n {classification_tasks}"
        return regression_prompt, classification_prompt


class PromptBuilder:
    def __init__(self):
        self.cfg = Config()
        self.prompt_creator = PromptBases()

    def format_pool_prompts(self, prompt_samples):
        """
        Obtain pool prompts from the model for classification.
        """
        # Format the pool prompts into a usable format
        pool_text = ""
        for idx, task in enumerate(prompt_samples["instruction"], 1):
            pool_text += f"Task {idx}: {task}\n"

        pool_input_prompt = self.prompt_creator.create_pool_task_prompt(pool_text, self.cfg)

        return pool_input_prompt


    def format_classification_classification(self):
        pass

    def format_instance_generation(self, task_classification, task_regression):
        """
        Takes in regression and classification tasks -> generates:
        instances for regression
        inputs for classification
        """

class ModelParser:
    def __init__(self):
        pass

    def get_pool_tasks(self, human_tasks, llm_tasks, total_needed):
        """Sample seed tasks with ratio of human and llm generated tasks"""

        # Determine the split of human to model prompts
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

        # Extract the sampled tasks for the pipeline
        human_pool_prompts = [task["instruction"] for task in human_sample_indices]
        llm_pool_prompts = [task["instruction"] for task in llm_sample_indices]

        return {
            "instruction": human_pool_prompts + llm_pool_prompts,
            "is_classification": [task["is_classification"] for task in human_sample_indices] + [task["is_classification"] for task in llm_sample_indices]
        }

    def format_classification_outputs(self, task_class_str):
        """
        Parse a string of tasks and their classification labels.
        Example input: "Task 1: Classification\nIs it classification? Yes\nTask 2: Regression\nIs it classification? No"
        Returns a list of dictionaries with 'id', 'task', and 'is_classification' keys.
        """
        task_keyword = pp.CaselessLiteral("Task")
        task_id = pp.pyparsing_common.integer('id')
        colon = pp.Suppress(':')
        task_body = pp.restOfLine("task_description")

        class_keyword = pp.CaselessLiteral("Is it classification?")
        class_value = pp.one_of('Yes No', caseless=True)('is_cls')

        task_line = task_keyword + task_id + colon + task_body + pp.LineEnd()
        class_line = class_keyword + class_value + pp.LineEnd()
        task_block = task_line + class_line

        # Parse all task blocks
        regression_tasks = []
        classification_tasks = []
        for tokens, start, end in task_block.scanString(task_class_str):
            if tokens.is_cls.lower() == 'yes':
                classification_tasks.append({'task': tokens.task_description.strip()})
            elif tokens.is_cls.lower() == 'no':
                regression_tasks.append({'task': tokens.task_description.strip()})
            else:
                print(f"Could not filter for id: {tokens.task_id}")

        print("Classification tasks:", classification_tasks)
        print("Regression tasks:", regression_tasks)

        return classification_tasks, regression_tasks


    def split_tasks_by_classification(self, tasks):
        """
        Splits parsed tasks into classification and non-classification dictionaries.

        Args:
            tasks: List of dicts from parse_task_class, each with 'id', 'task', 'is_classification'

        Returns:
            Tuple of (classification_dict, nonclassification_dict) with ID as key and task as value
        """
        classification_tasks = []
        regression_tasks = []

        for task in tasks:
            if task['is_classification']:
                classification_tasks.append(task['task'])
            else:
                regression_tasks.append(task['task'])

        return classification_tasks, regression_tasks


class PoolFilter:
    def __init__(self):
        self.cfg = Config()

    def filter_bad_words(self, prompts):
        """
        Iterates over prompts, excluding ones that contain 'bad words'
        """
        return [p for p in prompts if not any(bw in p for bw in self.cfg.bad_words)]

    def filter_length(self, prompts):
        return [p for p in prompts if len(p) >= self.cfg.len_threshold]

    def filter_ROUGE(self, prompts, reference):
        rouge = evaluate.load('rouge')
        filtered = []
        for p in prompts:
            result = rouge.compute(predictions=p, references=reference)
            if result['rougeL'] < self.cfg.rouge_threshold:
                filtered.append(p)
        return filtered

    def filter_tasks(self, prompts, pool_tasks):
        prompts = self.filter_bad_words(prompts)
        prompts = self.filter_length(prompts)
        prompts = self.filter_ROUGE(prompts, pool_tasks)
        return prompts
