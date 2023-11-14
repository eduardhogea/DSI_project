from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from feature_extraction import extract_features
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(documents, labels):
    # Extract features
    features, _ = extract_features(documents)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

    # Training the model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluating the model
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    return model, X_test, y_test, predictions

def plot_confusion_matrix(y_true, y_pred, categories):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting using seaborn
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=categories, yticklabels=categories)
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi = 300)

if __name__ == "__main__":
    # Loading the dataset
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups_train.data
    labels = newsgroups_train.target
    categories = newsgroups_train.target_names

    model, X_test, y_test, predictions = train_model(documents, labels)
    plot_confusion_matrix(y_test, predictions, categories)
