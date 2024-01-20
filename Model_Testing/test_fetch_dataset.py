import unittest
from unittest.mock import patch, mock_open, MagicMock
from Data_Preprocessing_and_Analysis import fetch_dataset
import json

class TestFetchDataset(unittest.TestCase):

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({
        'documents': ['document1', 'document2'],
        'categories': ['category1', 'category2']
    }))
    def test_fetch_data_local(self, mock_open, mock_exists):
        """
        Test fetch_data when the data is already saved locally.
        """
        mock_exists.return_value = True  # Simulate that the file exists

        docs, cats = fetch_dataset.fetch_data()

        self.assertEqual(docs, ['document1', 'document2'])
        self.assertEqual(cats, ['category1', 'category2'])
        mock_open.assert_called_once_with(fetch_dataset.DATA_PATH, 'r', encoding='utf-8')

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('Data_Preprocessing_and_Analysis.fetch_dataset.fetch_20newsgroups')
    def test_fetch_data_download(self, mock_fetch_20newsgroups, mock_open, mock_exists):
        """
        Test fetch_data when the data is not available locally and needs to be downloaded.
        """
        mock_exists.return_value = False  # Simulate that the file does not exist
        mock_fetch_20newsgroups.return_value = MagicMock(data=['document3', 'document4'], target_names=['category3', 'category4'])

        docs, cats = fetch_dataset.fetch_data()

        self.assertEqual(docs, ['document3', 'document4'])
        self.assertEqual(cats, ['category3', 'category4'])
        mock_open.assert_called_with(fetch_dataset.DATA_PATH, 'w', encoding='utf-8')
        mock_fetch_20newsgroups.assert_called_once_with(subset='train', remove=('headers', 'footers', 'quotes'))

if __name__ == '__main__':
    unittest.main()