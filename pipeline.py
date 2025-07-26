import random
from model import LM
from text_processing import JsonUtils, PromptBases, PoolFilter, Config, ModelParser, PromptBuilder

class Pipeline:
    """
    Main pipeline for the self-instruct paper
    Includes methods for the primary pipeline, including parsing of model outputs
    """
    def __init__(self):
        self.json_utils = JsonUtils("seed_tasks.jsonl")
        self.prompt_builder = PrompBuilder()
        self.pool_filter = PoolFilter()
        self.model_parser = ModelParser()
        self.prompt_bases = PromptBases()
        self.llm_seed_tasks = []
        self.human_seed_tasks = self.json_utils.load_data()
        self.model = LM()
        self.cfg = Config()
            
    def run_pipeline(self):
        for i in range(self.cfg.max_iterations):

            # Step 1a: Get examples for task generation
            sample_tasks = self.model_parser.get_pool_tasks(
                self.human_seed_tasks,
                self.llm_seed_tasks,
                self.cfg.num_seed_tasks,
            )

            # Step 1b: Format the examples in a useful format
            formatted_pool_tasks = self.prompt_builder.format_pool_prompts(sample_tasks)
            # Step 1c: Run the model to produce new tamodel_formattersks
            output_tasks = self.model.run_model(formatted_pool_tasks)


            # Step 2a: Create basic classification prompt for LLM to understand the goal using in-context learning
            human_labeled_classification_input_prompt = self.prompt_bases.create_classification_prompt(sample_tasks)

            # Step 2b: Join the example prompt with the LLM outputs
            if output_tasks:
                classification_input_prompt = (human_labeled_classification_input_prompt + "\n" + output_tasks)
            else:
                print("No LLM generated tasks, there is an error")

            # Step 2c: Run the model to generate "Task, Is classification" format
            output_classes = self.model.run_model(classification_input_prompt)

            # Step 3a: Parse the output of the tasks to return the two task types
            classification_tasks, regression_tasks = self.model_parser.format_classification_outputs(output_classes)


            # Step 3c: Run the model to generate examples using the class method
            regression_instance_prompt, classification_prompt = self.prompt_bases.create_instance_generation_prompt(regression_tasks, classification_tasks)

            # Step 4: Instance generation
            instances = self.
