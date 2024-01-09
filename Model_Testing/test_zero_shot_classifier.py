import unittest
import numpy as np
from Model_Development import ZeroShotClassifier, classify

class TestZeroShotClassifier(unittest.TestCase):

    def setUp(self):
        self.zero_shot_classifier = ZeroShotClassifier()
        self.sample_text = "This is a test sentence for classification."
        self.sample_labels = ["Label1", "Label2", "Label3"]

    def test_classify(self):
        result = self.zero_shot_classifier.classify(self.sample_text, self.sample_labels)
        self.assertIsInstance(result, dict)
        self.assertTrue("labels" in result)
        self.assertTrue("scores" in result)

    def test_zero_shot_classifier_load_model(self):
        loaded_classifier = self.zero_shot_classifier.load_model()
        self.assertIsNotNone(loaded_classifier)

class TestZeroShotClassifierService(unittest.TestCase):

    def setUp(self):
        self.sample_text = "This is a test sentence for classification."
        self.sample_labels = ["Label1", "Label2", "Label3"]

    def test_classify(self):
        input_data = {
            "text": self.sample_text,
            "candidate_labels": self.sample_labels
        }
        result = classify(input_data)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, len(self.sample_labels)))

if __name__ == "__main__":
    unittest.main()
