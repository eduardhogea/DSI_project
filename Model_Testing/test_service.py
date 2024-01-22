import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from Model_Development.service import classify_distilbert, classify_mobilebert, classify_roberta_large, \
    classify_squeezebert, ClassificationInput


class TestBentoMLService(unittest.TestCase):

    @patch('Model_Development.service.ZeroShotClassifier.classify')
    def test_classify_distilbert_endpoint(self, mock_classify):
        # Setup mock return value
        mock_return_value = {'labels': ['label1', 'label2'], 'scores': [0.9, 0.1]}
        mock_classify.return_value = mock_return_value

        # Create a sample input
        sample_input = ClassificationInput(text="Test text", candidate_labels=["label1", "label2"])

        # Call the classify_distilbert endpoint
        response = classify_distilbert(sample_input)

        # Assertions
        mock_classify.assert_called_with("Test text", ["label1", "label2"])
        self.assertIsInstance(response, np.ndarray)
        self.assertEqual(response[0].tolist(), ['label1', 'label2'])

        # Convert the list of floats to strings before comparing
        self.assertEqual([str(x) for x in response[1].tolist()], ['0.9', '0.1'])

    # Similar tests for other endpoints (classify_mobilebert, classify_roberta_large, classify_squeezebert)


if __name__ == '__main__':
    unittest.main()
