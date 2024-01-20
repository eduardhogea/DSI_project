import unittest
from Feature_Engineering.feature_extraction import extract_features  # Import the function, not the module
from sklearn.feature_extraction.text import TfidfVectorizer

class TestFeatureExtraction(unittest.TestCase):

    def test_feature_extraction_output_type(self):
        """ Test if the output is a tuple containing the feature matrix and vectorizer """
        sample_docs = ["Sample document text.", "Another sample document."]
        output = extract_features(sample_docs)  # Correctly call the function here
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 2)
        self.assertIsInstance(output[0], type(TfidfVectorizer().fit_transform(sample_docs)))
        self.assertIsInstance(output[1], TfidfVectorizer)

    def test_feature_extraction_shape(self):
        """ Test if the feature matrix shape matches the expected shape """
        sample_docs = ["Sample document text.", "Another sample document."]
        feature_matrix, _ = extract_features(sample_docs)  # Correctly call the function here
        self.assertEqual(feature_matrix.shape[0], len(sample_docs))
        # More specific shape tests can be added depending on the preprocessing and vectorization logic

if __name__ == '__main__':
    unittest.main()