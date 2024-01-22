"""
Module to fetch and save the 20 Newsgroups dataset.
This module checks for a local copy of the dataset first and downloads it if not available.
It saves the dataset locally for future use.
"""

"""
Testing something here for git.

"""

import os
import json
from sklearn.datasets import fetch_20newsgroups

# Path for saving the dataset
DATA_PATH = "newsgroups_data.json"

def fetch_data():
    """
    Fetches the 20 Newsgroups dataset.
    
    Checks for a local copy of the dataset and loads it if available.
    Otherwise, it downloads the dataset and saves it locally.

    Returns:
    tuple: A tuple containing the list of documents and their corresponding categories.
    """
    # Check if the data is already saved locally
    if os.path.exists(DATA_PATH):
        print("Loading data from local file...")
        with open(DATA_PATH, 'r', encoding='utf-8') as file:
            local_data = json.load(file)
        docs = local_data['documents']
        cats = local_data['categories']
    else:
        print("Downloading the dataset...")
        newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        docs = newsgroups_train.data
        cats = newsgroups_train.target_names

        # Save the data locally
        with open(DATA_PATH, 'w', encoding='utf-8') as file:
            json.dump({'documents': docs, 'categories': cats}, file)

    return docs, cats

if __name__ == "__main__":
    documents, categories = fetch_data()
    print("Sample document:", documents[0])
    print("Categories:", categories)
