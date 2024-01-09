"""
Module for creating and managing a Zero Shot Classifier model.
Includes class definition, model saving and loading functions.
"""

import os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import bentoml

class ZeroShotClassifier:
    """
    Zero Shot Classifier using Hugging Face's transformers library.

    Attributes:
    model_name (str): The name of the model to use.
    save_dir (str): Directory to save the model.
    classifier (transformers.Pipeline): The classifier pipeline.
    """
    def __init__(self, model_name="typeform/distilbert-base-uncased-mnli", save_dir="model"):
        """
        Initializes the ZeroShotClassifier with a model name and save directory.
        """
        self.model_name = model_name
        self.save_dir = save_dir
        self.classifier = self.load_model()

    def load_model(self):
        """
        Loads the model from the specified directory, or downloads it if not present.

        Returns:
        transformers.Pipeline: The zero-shot classification pipeline.
        """
        if os.path.exists(self.save_dir):
            model = AutoModelForSequenceClassification.from_pretrained(self.save_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model.save_pretrained(self.save_dir)
            tokenizer.save_pretrained(self.save_dir)
        return pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    def classify(self, text, candidate_labels):
        """
        Classifies the given text against a set of candidate labels.

        Args:
        text (str): The text to classify.
        candidate_labels (list of str): The candidate labels for classification.

        Returns:
        dict: Classification results with labels and scores.
        """
        return self.classifier(text, candidate_labels)

def save_model(classifier_instance, model_name):
    """
    Saves the model to the specified directory.

    Args:
    classifier_instance (ZeroShotClassifier): The classifier instance to save.
    model_name (str): The name to save the model under.
    """
    bentoml.transformers.save_model(model_name, classifier_instance.classifier.model, classifier_instance.classifier.tokenizer)

def load_model(model_name):
    """
    Loads the model with the given name.

    Args:
    model_name (str): The name of the model to load.

    Returns:
    transformers.Pipeline: The loaded zero-shot classification pipeline.
    """
    classifier = bentoml.transformers.load_model(model_name)
    return classifier

if __name__ == "__main__":
    zero_shot_classifier = ZeroShotClassifier()

    save_model(zero_shot_classifier, "my_zero_shot_classifier")

    loaded_classifier = load_model("my_zero_shot_classifier")
    print("Model loaded successfully.")
