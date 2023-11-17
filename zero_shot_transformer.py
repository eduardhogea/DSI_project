from transformers import pipeline
from sklearn.datasets import fetch_20newsgroups

def zero_shot_classification(documents, categories):
    classifier = pipeline("zero-shot-classification")
    
    for doc in documents[:5]:  # Limiting to first 5 documents for demonstration
        result = classifier(doc, candidate_labels=categories)
        print(f"Document: {doc[:100]}...")  # Displaying first 100 characters of the document
        print("Predicted Category:", result['labels'][0], "\n")  # Top predicted category

if __name__ == "__main__":
    # Load the 20 Newsgroups dataset
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups_train.data
    categories = newsgroups_train.target_names

    # Perform zero-shot classification
    zero_shot_classification(documents, categories)
