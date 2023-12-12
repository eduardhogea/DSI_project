import sys
import os

script_dir = os.path.dirname(__file__)                             # The directory of the current script
parent_dir = os.path.dirname(script_dir)                           # Path to the parent directory
feature_eng_dir = os.path.join(parent_dir, "Feature_Engineering")  # Path to the "Feature Engineering" directory
images_dir = os.path.join(parent_dir, "Images")                    # Path to the "Images" directory
sys.path.append(feature_eng_dir)                                   # Append the directory to sys.path


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from feature_extraction import extract_features
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(documents, labels, model_type='naive_bayes'):
    # Extract features
    features, _ = extract_features(documents)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

    # Select and train the model based on the specified type
    if model_type == "naive_bayes":
        model = MultinomialNB()
    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "linear_svc":
        model = LinearSVC(max_iter=1000, dual=False)
    else:
        raise ValueError("Unsupported model type")


    model.fit(X_train, y_train)

    # Evaluating the model
    predictions = model.predict(X_test)
    print(f"Result for {model_type}:")
    print(classification_report(y_test, predictions))

    return model, X_test, y_test, predictions


def plot_confusion_matrix(y_true, y_pred, categories, filename):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plotting using seaborn
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=categories, yticklabels=categories)
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # Save using the provided filename
    plt.close()


if __name__ == "__main__":
    # Loading the dataset
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups_train.data
    labels = newsgroups_train.target
    categories = newsgroups_train.target_names

    # Train and evaluate Naive Bayes
    nb_model, X_test, y_test, nb_predictions = train_model(documents, labels, "naive_bayes")
    confusion_matrix_path = os.path.join(images_dir, "confusion_matrix_naive_bayes.png")
    plot_confusion_matrix(y_test, nb_predictions, categories, confusion_matrix_path)

    # Train and evaluate Logistic Regression
    lr_model, _, _, lr_predictions = train_model(documents, labels, "logistic_regression")
    confusion_matrix_path = os.path.join(images_dir, "confusion_matrix_logistic_regression.png")
    plot_confusion_matrix(y_test, lr_predictions, categories, confusion_matrix_path)

    # Train and evaluate Linear SVM
    svm_model, _, _, svm_predictions = train_model(documents, labels, "linear_svc")
    confusion_matrix_path = os.path.join(images_dir, "confusion_matrix_linear_svc.png")
    plot_confusion_matrix(y_test, svm_predictions, categories, confusion_matrix_path)