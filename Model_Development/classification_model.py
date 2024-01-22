"""
Module for developing and evaluating classification models on text data.
It includes functions for training models and plotting confusion matrices.

Testing something here for git.
"""

import sys
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from Feature_Engineering.feature_extraction import extract_features
from sklearn.datasets import fetch_20newsgroups

# Directory setup
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
feature_eng_dir = os.path.join(parent_dir, "Feature_Engineering")
images_dir = os.path.join(parent_dir, "Images")
sys.path.append(feature_eng_dir)

def train_model(docs, lbls, model_type='naive_bayes'):
    """
    Trains a specified model on the given documents and labels.

    Args:
    docs (list of str): The documents to train on.
    lbls (list): The labels for the documents.
    model_type (str): The type of model to train. Options are 'naive_bayes', 'logistic_regression', 'linear_svc'.

    Returns:
    tuple: The trained model, test features, test labels, and predictions.
    """
    # Extract features
    features, _ = extract_features(docs)

    # Splitting the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, lbls, test_size=0.2, random_state=1)

    # Select and train the model
    if model_type == "naive_bayes":
        model = MultinomialNB()
    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "linear_svc":
        model = LinearSVC(max_iter=1000, dual=False)
    else:
        raise ValueError("Unsupported model type")

    model.fit(x_train, y_train)

    # Evaluating the model
    predictions = model.predict(x_test)
    print(f"Result for {model_type}:")
    print(classification_report(y_test, predictions))

    return model, x_test, y_test, predictions

def plot_confusion_matrix(y_true, y_pred, categories, filename):
    """
    Plots a confusion matrix for the given true and predicted labels.

    Args:
    y_true (list): The true labels.
    y_pred (list): The predicted labels.
    categories (list of str): Category names for the labels.
    filename (str): Path to save the plot.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plotting using seaborn
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=categories, yticklabels=categories)
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Loading the dataset
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    docs = newsgroups_train.data
    lbls = newsgroups_train.target
    cats = newsgroups_train.target_names

    # Train and evaluate models
    nb_model, x_test_nb, y_test_nb, nb_predictions = train_model(docs, lbls, "naive_bayes")
    nb_cm_path = os.path.join(images_dir, "confusion_matrix_naive_bayes.png")
    plot_confusion_matrix(y_test_nb, nb_predictions, cats, nb_cm_path)

    lr_model, x_test_lr, y_test_lr, lr_predictions = train_model(docs, lbls, "logistic_regression")
    lr_cm_path = os.path.join(images_dir, "confusion_matrix_logistic_regression.png")
    plot_confusion_matrix(y_test_lr, lr_predictions, cats, lr_cm_path)

    svm_model, x_test_svm, y_test_svm, svm_predictions = train_model(docs, lbls, "linear_svc")
    svm_cm_path = os.path.join(images_dir, "confusion_matrix_linear_svc.png")
    plot_confusion_matrix(y_test_svm, svm_predictions, cats, svm_cm_path)
