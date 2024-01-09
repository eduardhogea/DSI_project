"""
Module for creating and serving a machine learning model using BentoML.
It includes a ZeroShotClassifier class and an API endpoint for model inference.
"""

import os
import numpy as np
import bentoml
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

class ZeroShotClassifier:
    """
    Classifier for zero-shot learning using Hugging Face's transformers.

    Attributes:
    model_name (str): Name of the model to be used.
    save_dir (str): Directory to save the model.
    classifier (transformers.Pipeline): The loaded classification model.
    """
    def __init__(self, model_name="typeform/distilbert-base-uncased-mnli", save_dir="model"):
        """
        Initialize the ZeroShotClassifier with a model name and save directory.
        """
        self.model_name = model_name
        self.save_dir = save_dir
        self.classifier = self.load_model()

    def load_model(self):
        """
        Load the model from the specified directory or download it if not present.
        
        Returns:
        transformers.Pipeline: The loaded zero-shot classification pipeline.
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
        Classify the given text against a set of candidate labels.

        Args:
        text (str): The text to classify.
        candidate_labels (list of str): The candidate labels for classification.

        Returns:
        dict: Classification results with labels and scores.
        """
        return self.classifier(text, candidate_labels)

# Load your ZeroShotClassifier instance
zero_shot_classifier = ZeroShotClassifier()

class ClassificationInput(BaseModel):
    """
    Pydantic model for input validation in the API endpoint.

    Attributes:
    text (str): The text to classify.
    candidate_labels (list of str): The candidate labels for classification.
    """
    text: str
    candidate_labels: list[str]

input_spec = JSON(pydantic_model=ClassificationInput)

# Initialize BentoML service
svc = bentoml.Service("zero_shot_classifier_service", runners=[])

@svc.api(input=input_spec, output=NumpyNdarray())
def classify(input_data: ClassificationInput) -> np.ndarray:
    """
    API endpoint for classifying text using the ZeroShotClassifier.

    Args:
    input_data (ClassificationInput): The input data with text and candidate labels.

    Returns:
    np.ndarray: The classification results including labels and scores.
    """
    text = str(input_data.text)
    candidate_labels = input_data.candidate_labels

    # Use the ZeroShotClassifier's classify method for inference
    result = zero_shot_classifier.classify(text, candidate_labels)

    # Assuming result is a dictionary with keys 'labels' and 'scores'
    labels = result.get('labels', [])
    scores = result.get('scores', [])
    
    return np.array([labels, scores])
