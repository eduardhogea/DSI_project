import os
import json
from sklearn.datasets import fetch_20newsgroups

# Path for saving the dataset
DATA_PATH = "newsgroups_data.json"

def fetch_data():
    # Check if the data is already saved locally
    if os.path.exists(DATA_PATH):
        print("Loading data from local file...")
        with open(DATA_PATH, 'r') as file:
            data = json.load(file)
        documents = data['documents']
        categories = data['categories']
    else:
        print("Downloading the dataset...")
        newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        documents = newsgroups_train.data
        categories = newsgroups_train.target_names

        # Save the data locally
        with open(DATA_PATH, 'w') as file:
            json.dump({'documents': documents, 'categories': categories}, file)

    return documents, categories

if __name__ == "__main__":
    documents, categories = fetch_data()
    print("Sample document:", documents[0])
    print("Categories:", categories)
