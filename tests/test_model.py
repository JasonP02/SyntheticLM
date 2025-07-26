import unittest
from model import LM

class TestLM(unittest.TestCase):
    def test_init_with_api_key(self):
        """Test LM initialization with provided API key"""
        lm = LM("test-key")
        self.assertEqual(lm.client.api_key, "test-key")
    
    def test_init_without_api_key(self):
        """Test LM initialization without API key (reads from file)"""
        # This test would require mocking file reading
        pass

if __name__ == '__main__':
    unittest.main()