from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import os

class ZeroShotClassifier:
    def __init__(self, model_name="typeform/distilbert-base-uncased-mnli", save_dir="model"):
        self.model_name = model_name
        self.save_dir = save_dir
        self.classifier = self.load_model()

    def load_model(self):
        if os.path.exists(self.save_dir):
            # Load model from saved directory
            model = AutoModelForSequenceClassification.from_pretrained(self.save_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        else:
            # Download and save the model and tokenizer
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model.save_pretrained(self.save_dir)
            tokenizer.save_pretrained(self.save_dir)

        return pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    def classify(self, text, candidate_labels):
        return self.classifier(text, candidate_labels)
