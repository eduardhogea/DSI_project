from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import bentoml
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel
import numpy as np
import os

# Your ZeroShotClassifier definition
class ZeroShotClassifier:
    def __init__(self, model_name="typeform/distilbert-base-uncased-mnli", save_dir="model"):
        self.model_name = model_name
        self.save_dir = save_dir
        self.classifier = self.load_model()

    def load_model(self):
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
        return self.classifier(text, candidate_labels)

# Load your ZeroShotClassifier instance
zero_shot_classifier = ZeroShotClassifier()

# Define the Pydantic model for input validation
class ClassificationInput(BaseModel):
    text: str
    candidate_labels: list[str]

input_spec = JSON(pydantic_model=ClassificationInput)

# Initialize BentoML service
svc = bentoml.Service("zero_shot_classifier_service", runners=[])

@svc.api(input=input_spec, output=NumpyNdarray())
def classify(input_data: ClassificationInput) -> np.ndarray:
    text = str(input_data.text)
    candidate_labels = input_data.candidate_labels

    # Use the ZeroShotClassifier's classify method for inference
    result = zero_shot_classifier.classify(text, candidate_labels)

    # Assuming result is a dictionary with keys 'labels' and 'scores'
    labels = result.get('labels', [])
    scores = result.get('scores', [])
    
    return np.array([labels, scores])

