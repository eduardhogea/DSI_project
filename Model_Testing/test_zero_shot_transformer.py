import unittest
import os
import sys
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase, AddedToken
from Model_Development.zero_shot_transformer import ZeroShotClassifier, save_model, load_model

class TestZeroShotClassifier(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = 'test_model'
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        # Clean up the temporary directory after testing
        os.rmdir(self.test_dir)

    def test_classifier_initialization(self):
        classifier = ZeroShotClassifier()
        self.assertIsNotNone(classifier)

    def test_classify_text(self):
        classifier = ZeroShotClassifier()
        text = "This is a test text."
        candidate_labels = ["Label1", "Label2", "Label3"]
        results = classifier.classify(text, candidate_labels)
        # Modify the assertion to check the format of results
        self.assertIsInstance(results, dict)
        self.assertTrue('sequence' in results)
        self.assertTrue('labels' in results)
        self.assertTrue('scores' in results)
        self.assertEqual(len(results['labels']), len(candidate_labels))


    def test_save_and_load_model(self):
        classifier = ZeroShotClassifier()
        model_name = "test_zero_shot_classifier"
        save_model(classifier, model_name)
        loaded_classifier = load_model(model_name, save_dir=self.test_dir)
        self.assertIsNotNone(loaded_classifier)

if __name__ == '__main__':
    unittest.main()
