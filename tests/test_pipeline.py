import unittest
from unittest.mock import patch, MagicMock
from pipeline import Pipeline

class TestPipeline(unittest.TestCase):
    @patch('pipeline.DataProcessor')
    @patch('pipeline.PromptBases')
    @patch('pipeline.PoolFilter')
    @patch('pipeline.ModelParser')
    @patch('pipeline.LM')
    @patch('pipeline.Config')
    def setUp(self, mock_config, mock_lm, mock_model_parser, mock_pool_filter, mock_prompt_bases, mock_data_processor):
        """Set up test pipeline with mocked dependencies"""
        # Create mock instances
        self.mock_data_processor = MagicMock()
        self.mock_prompt_bases = MagicMock()
        self.mock_pool_filter = MagicMock()
        self.mock_model_parser = MagicMock()
        self.mock_lm = MagicMock()
        self.mock_config = MagicMock()
        
        # Configure mocks
        mock_data_processor.return_value = self.mock_data_processor
        mock_prompt_bases.return_value = self.mock_prompt_bases
        mock_pool_filter.return_value = self.mock_pool_filter
        mock_model_parser.return_value = self.mock_model_parser
        mock_lm.return_value = self.mock_lm
        mock_config.return_value = self.mock_config
        
        # Create pipeline instance
        self.pipeline = Pipeline()
    
    def test_init(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline.data_processor)
        self.assertIsNotNone(self.pipeline.model_formatter)
        self.assertIsNotNone(self.pipeline.pool_filter)
        self.assertIsNotNone(self.pipeline.model_parser)
        self.assertIsNotNone(self.pipeline.prompt_bases)
        self.assertIsNotNone(self.pipeline.model)
        self.assertIsNotNone(self.pipeline.cfg)

if __name__ == '__main__':
    unittest.main()