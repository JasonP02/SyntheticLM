import unittest
import json
from text_processing import DataProcessor, PromptBases, Config, ModelParser

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.test_data = [
            {
                "id": "test_0",
                "name": "test_task",
                "instruction": "This is a test instruction",
                "instances": [{"input": "", "output": "Test output"}],
                "is_classification": False
            }
        ]
        
        # Write test data to temporary file
        with open("test_data.jsonl", "w") as f:
            for item in self.test_data:
                f.write(json.dumps(item) + "\n")
        
        self.processor = DataProcessor("test_data.jsonl")
    
    def tearDown(self):
        """Clean up test files"""
        import os
        if os.path.exists("test_data.jsonl"):
            os.remove("test_data.jsonl")
    
    def test_load_data(self):
        """Test loading data from JSONL file"""
        data = self.processor.load_data()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["id"], "test_0")
    
    def test_save_data_as_jsonl(self):
        """Test saving data to JSONL file"""
        test_output = [{"id": "output_0", "value": "test"}]
        self.processor.save_data_as_jsonl(test_output, "test_output.jsonl")
        
        # Verify the file was created and contains correct data
        with open("test_output.jsonl", "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            saved_data = json.loads(lines[0])
            self.assertEqual(saved_data["id"], "output_0")
        
        # Clean up
        import os
        if os.path.exists("test_output.jsonl"):
            os.remove("test_output.jsonl")

class TestPromptBases(unittest.TestCase):
    def setUp(self):
        self.prompt_bases = PromptBases()
        self.config = Config()
    
    def test_create_pool_task_prompt(self):
        """Test creating pool task prompt"""
        pool_text = "Task 1: Do something\nTask 2: Do something else"
        prompt = self.prompt_bases.create_pool_task_prompt(pool_text, self.config)
        self.assertIn("Come up with a series of tasks", prompt)
        self.assertIn(pool_text, prompt)
    
    def test_create_classification_prompt(self):
        """Test creating classification prompt"""
        prompt_dict = {
            "instruction": ["Task 1", "Task 2"],
            "is_classification": [True, False]
        }
        prompt = self.prompt_bases.create_classification_prompt(prompt_dict)
        self.assertIn("Can the following task be regarded as a classification task", prompt)
        self.assertIn("Task 1", prompt)
        self.assertIn("Yes", prompt)
        self.assertIn("Task 2", prompt)
        self.assertIn("No", prompt)
    
    def test_create_instance_generation_prompt(self):
        """Test creating instance generation prompt"""
        regression_tasks = ["Task 1", "Task 2"]
        classification_tasks = ["Task 3", "Task 4"]
        reg_prompt, class_prompt = self.prompt_bases.create_instance_generation_prompt(
            regression_tasks, classification_tasks
        )
        self.assertIn("Come up with examples for the following tasks", reg_prompt)
        self.assertIn("generate an input that corresponds to each of the class labels", class_prompt)

class TestModelParser(unittest.TestCase):
    def setUp(self):
        self.parser = ModelParser()
    
    def test_get_pool_tasks(self):
        """Test sampling pool tasks"""
        human_tasks = [
            {"instruction": "Task 1", "is_classification": True},
            {"instruction": "Task 2", "is_classification": False},
            {"instruction": "Task 3", "is_classification": True}
        ]
        llm_tasks = []
        result = self.parser.get_pool_tasks(human_tasks, llm_tasks, 2)
        self.assertEqual(len(result["instruction"]), 2)
        self.assertEqual(len(result["is_classification"]), 2)
    
    def test_parse_task_class(self):
        """Test parsing task class string"""
        task_class_str = """Task 1: Do something
Is it classification? Yes
Task 2: Do something else
Is it classification? No"""
        result = self.parser.parse_task_class(task_class_str)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[0]["task"], "Do something")
        self.assertEqual(result[0]["is_classification"], True)
        self.assertEqual(result[1]["id"], 2)
        self.assertEqual(result[1]["task"], "Do something else")
        self.assertEqual(result[1]["is_classification"], False)
    
    def test_split_tasks_by_classification(self):
        """Test splitting tasks by classification"""
        tasks = [
            {"id": 1, "task": "Task 1", "is_classification": True},
            {"id": 2, "task": "Task 2", "is_classification": False},
            {"id": 3, "task": "Task 3", "is_classification": True}
        ]
        classification_tasks, regression_tasks = self.parser.split_tasks_by_classification(tasks)
        self.assertEqual(len(classification_tasks), 2)
        self.assertEqual(len(regression_tasks), 1)

if __name__ == '__main__':
    unittest.main()