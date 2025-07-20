import random
from model import LM
from text_processing import DataProcessor, PromptCreator, PoolFilter, Config

class Pipeline:
    def __init__(self):
        self.data_processor = DataProcessor("seed_tasks.jsonl")
        self.prompt_creator = PromptCreator()
        self.pool_filter = PoolFilter()
        self.llm_seed_tasks = []
        self.human_seed_tasks = self.data_processor.load_data()
        self.model = LM()
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

            # Step 1: Generate pool tasks from human and LLM seed tasks
            formatted_pool_tasks = self.format_pool_prompts(sample_tasks)
            output_tasks = self.model.run_model(formatted_pool_tasks)

            # Step 2: Classify the generated tasks as classification or regression
            classification_input_prompt = self.prompt_creator.create_classification_prompt(sample_tasks)
            output_classes = self.model.run_model(classification_input_prompt + (output_tasks or ""))
            print(output_classes)
            
            # Step 3: Generate data based on the task type
            # First, we need to split up the output_classes based on whether they are classification or not, then feed into model with distinct prompts

            parsed_class_data = self.data_processor.parse_task_class(classification_input_prompt, output_classes)
            print(parsed_class_data)

            # parsed_class_data is a list of dicts: [{"task": ..., "class": ...}, ...]

            classification_tasks = [item["task"] for item in parsed_class_data if item["class"].lower().startswith("y")]
            regression_tasks = [item["task"] for item in parsed_class_data if item["class"].lower().startswith("n")]

            print("Classification tasks:", classification_tasks)
            print("Regression tasks:", regression_tasks)