import unittest
import os
from Model_Development.classification_model import train_model, plot_confusion_matrix
from sklearn.datasets import fetch_20newsgroups

class TestClassificationModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Loading a sample dataset for testing
        cls.newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=['alt.atheism', 'sci.space'])
        cls.docs = cls.newsgroups_train.data
        cls.lbls = cls.newsgroups_train.target
        cls.cats = cls.newsgroups_train.target_names

    def test_classification_model_output(self):
        model, x_test, y_test, predictions = train_model(self.docs, self.lbls, "naive_bayes")
        self.assertIsNotNone(model)
        self.assertIsNotNone(x_test)
        self.assertIsNotNone(y_test)
        self.assertIsNotNone(predictions)

    def test_classification_model_invalid_type(self):
        with self.assertRaises(ValueError):
            train_model(self.docs, self.lbls, "invalid_model_type")

    def test_classification_model(self):
        _, _, y_test, predictions = train_model(self.docs, self.lbls, "naive_bayes")
        output_file = "test_confusion_matrix.png"
        plot_confusion_matrix(y_test, predictions, self.cats, output_file)
        self.assertTrue(os.path.isfile(output_file))
        os.remove(output_file)  # Clean up

if _name_ == '_main_':
   unittest.main()