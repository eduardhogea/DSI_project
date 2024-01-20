import unittest
from unittest.mock import patch, MagicMock
from Model_Development.classification_model import train_model, plot_confusion_matrix
from sklearn.datasets import fetch_20newsgroups

class TestTextClassification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=['alt.atheism', 'sci.space'])
        cls.docs = cls.newsgroups_train.data
        cls.lbls = cls.newsgroups_train.target
        cls.cats = cls.newsgroups_train.target_names

    @patch('Model_Development.classification_model.extract_features')
    @patch('Model_Development.classification_model.train_test_split')
    @patch('Model_Development.classification_model.MultinomialNB')
    def test_train_model_naive_bayes(self, mock_nb, mock_split, mock_extract):
        mock_extract.return_value = (MagicMock(), MagicMock())
        mock_split.return_value = (MagicMock(), MagicMock(), MagicMock(), self.lbls)
        mock_model = MagicMock()
        mock_model.predict.return_value = self.lbls
        mock_nb.return_value = mock_model

        model, x_test, y_test, predictions = train_model(self.docs, self.lbls, "naive_bayes")

        mock_extract.assert_called_with(self.docs)
        mock_split.assert_called()
        mock_nb().fit.assert_called()
        self.assertIsNotNone(model)
        self.assertIsNotNone(predictions)

    @patch('Model_Development.classification_model.extract_features')
    @patch('Model_Development.classification_model.train_test_split')
    @patch('Model_Development.classification_model.LogisticRegression')
    def test_train_model_logistic_regression(self, mock_lr, mock_split, mock_extract):
        mock_extract.return_value = (MagicMock(), MagicMock())
        mock_split.return_value = (MagicMock(), MagicMock(), MagicMock(), self.lbls)
        mock_model = MagicMock()
        mock_model.predict.return_value = self.lbls
        mock_lr.return_value = mock_model

        model, x_test, y_test, predictions = train_model(self.docs, self.lbls, "logistic_regression")

        mock_extract.assert_called_with(self.docs)
        mock_split.assert_called()
        mock_lr().fit.assert_called()
        self.assertIsNotNone(model)
        self.assertIsNotNone(predictions)

    @patch('Model_Development.classification_model.extract_features')
    @patch('Model_Development.classification_model.train_test_split')
    @patch('Model_Development.classification_model.LinearSVC')
    def test_train_model_linear_svc(self, mock_svc, mock_split, mock_extract):
        mock_extract.return_value = (MagicMock(), MagicMock())
        mock_split.return_value = (MagicMock(), MagicMock(), MagicMock(), self.lbls)
        mock_model = MagicMock()
        mock_model.predict.return_value = self.lbls
        mock_svc.return_value = mock_model

        model, x_test, y_test, predictions = train_model(self.docs, self.lbls, "linear_svc")

        mock_extract.assert_called_with(self.docs)
        mock_split.assert_called()
        mock_svc().fit.assert_called()
        self.assertIsNotNone(model)
        self.assertIsNotNone(predictions)

    @patch('Model_Development.classification_model.plt.savefig')
    @patch('Model_Development.classification_model.confusion_matrix')
    @patch('Model_Development.classification_model.sns.heatmap')
    def test_plot_confusion_matrix(self, mock_heatmap, mock_cm, mock_savefig):
        y_true = self.lbls[:4]
        y_pred = self.lbls[:4]

        plot_confusion_matrix(y_true, y_pred, self.cats, "test_plot.png")

        mock_cm.assert_called_with(y_true, y_pred)
        mock_heatmap.assert_called()
        mock_savefig.assert_called_with("test_plot.png", dpi=300)

if __name__ == '__main__':
   unittest.main()