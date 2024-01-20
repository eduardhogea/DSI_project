import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from Model_Development.service import classify, ClassificationInput


class TestBentoMLService(unittest.TestCase):

    @patch('Model_Development.service.zero_shot_classifier.classify')
    def test_classify_endpoint(self, mock_classify):
        # Setup mock return value
        mock_return_value = {'labels': ['label1', 'label2'], 'scores': [0.9, 0.1]}
        mock_classify.return_value = mock_return_value

        # Create a sample input
        sample_input = ClassificationInput(text="Test text", candidate_labels=["label1", "label2"])

        # Call the classify endpoint
        response = classify(sample_input)

        # Assertions
        mock_classify.assert_called_with("Test text", ["label1", "label2"])
        self.assertIsInstance(response, np.ndarray)
        self.assertEqual(response[0].tolist(), ['label1', 'label2'])

        # Convert the list of floats to strings before comparing
        self.assertEqual([str(x) for x in response[1].tolist()], ['0.9', '0.1'])


if __name__ == '__main__':
   unittest.main()