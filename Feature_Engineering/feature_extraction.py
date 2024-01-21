"""
Module for feature extraction from text data using TF-IDF Vectorizer.
It includes a function to extract features from a given set of documents.
"""
"""
Testing something here for git.

"""

import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from Data_Preprocessing_and_Analysis.advanced_preprocess import advanced_preprocess_text
from sklearn.datasets import fetch_20newsgroups

# Update sys.path to include the target directory for imports
script_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
parent_dir = os.path.dirname(script_dir)  # Gets the parent directory
target_dir = os.path.join(parent_dir, "Data_Preprocessing_and_Analysis")  # Path to the target directory
sys.path.append(target_dir)

def extract_features(docs):
    """
    Extracts TF-IDF features from the provided documents.

    Args:
    docs (list of str): List of documents to extract features from.

    Returns:
    tuple: Tuple containing the feature matrix and the vectorizer.
    """
    vect = TfidfVectorizer(preprocessor=advanced_preprocess_text)
    feats = vect.fit_transform(docs)
    return feats, vect

if __name__ == "__main__":
    # Load a sample of the 20 Newsgroups dataset
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    docs = newsgroups_train.data

    # Extract features
    extracted_features, vectorizer = extract_features(docs[:100])  # Limiting to first 100 documents for demonstration
    print("Feature shape:", extracted_features.shape)
