import unittest
from Model_Development.zero_shot_transformer import ZeroShotClassifier, save_model, load_model

class TestZeroShotClassifier(unittest.TestCase):

    def setUp(self):
        self.zero_shot_classifier = ZeroShotClassifier()

    def test_classify(self):
        text = "This is a test sentence for classification."
        candidate_labels = ["Label1", "Label2", "Label3"]
        result = self.zero_shot_classifier.classify(text, candidate_labels)
        self.assertIsInstance(result, dict)
        self.assertTrue("labels" in result)
        self.assertTrue("scores" in result)

    def test_save_and_load_model(self):
        model_name = "my_zero_shot_classifier_test"
        save_model(self.zero_shot_classifier, model_name)
        loaded_classifier = load_model(model_name)
        
        self.assertIsInstance(loaded_classifier, ZeroShotClassifier)
        self.assertIsInstance(loaded_classifier.classifier, transformers.Pipeline)

if __name__ == "__main__":
    unittest.main()
