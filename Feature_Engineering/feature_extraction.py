import sys
import os

# we need to go up one level to the parent directory and then into the 'Data preprocessing and analysis' directory
script_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
parent_dir = os.path.dirname(script_dir)  # Gets the parent directory
target_dir = os.path.join(parent_dir, "Data_preprocessing_and_analysis")  # Path to the target directory
sys.path.append(target_dir)


from sklearn.feature_extraction.text import TfidfVectorizer
from advanced_preprocess import advanced_preprocess_text
from sklearn.datasets import fetch_20newsgroups

def extract_features(documents):
    vectorizer = TfidfVectorizer(preprocessor=advanced_preprocess_text)
    features = vectorizer.fit_transform(documents)
    return features, vectorizer

if __name__ == "__main__":
    # Load a sample of the 20 Newsgroups dataset
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups_train.data

    # Extract features
    features, vectorizer = extract_features(documents[:100])  # Limiting to first 100 documents for demonstration
    print("Feature shape:", features.shape)
