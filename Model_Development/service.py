"""
Testing something here for git.

"""

import os
import numpy as np
import bentoml
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

class ZeroShotClassifier:
    """
    A classifier for zero-shot learning using Hugging Face's transformers.
    This class supports different transformer models.

    Attributes:
        model_name (str): Name of the model to be used.
        save_dir (str): Directory to save the model.
        classifier (transformers.Pipeline): The loaded classification model.
    """
    def __init__(self, model_name, save_dir="model"):
        """
        Initialize the classifier with a specified transformer model.

        Args:
            model_name (str): The name of the transformer model to use.
            save_dir (str): The directory to save the model.
        """
        self.model_name = model_name
        self.save_dir = os.path.join(save_dir, model_name.replace('/', '_'))
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

# Pydantic model for input validation
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
svc = bentoml.Service("multi_model_classifier_service", runners=[])

# Helper function to classify text using a specified model
def classify_with_model(model_name, input_data):
    classifier = ZeroShotClassifier(model_name)
    text = str(input_data.text)
    candidate_labels = input_data.candidate_labels
    result = classifier.classify(text, candidate_labels)
    labels = result.get('labels', [])
    scores = result.get('scores', [])
    return np.array([labels, scores])

# Endpoints for different models
@svc.api(input=input_spec, output=NumpyNdarray())
def classify_distilbert(input_data: ClassificationInput) -> np.ndarray:
    return classify_with_model("typeform/distilbert-base-uncased-mnli", input_data)

@svc.api(input=input_spec, output=NumpyNdarray())
def classify_mobilebert(input_data: ClassificationInput) -> np.ndarray:
    return classify_with_model("typeform/mobilebert-uncased-mnli", input_data)

@svc.api(input=input_spec, output=NumpyNdarray())
def classify_roberta_large(input_data: ClassificationInput) -> np.ndarray:
    return classify_with_model("typeform/roberta-large-mnli", input_data)

@svc.api(input=input_spec, output=NumpyNdarray())
def classify_squeezebert(input_data: ClassificationInput) -> np.ndarray:
    return classify_with_model("typeform/squeezebert-mnli", input_data)
