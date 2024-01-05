import os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, TFAutoModelForSequenceClassification, PreTrainedTokenizerBase
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
        self.save_dir = os.path.join(save_dir, model_name.replace('/', '_'))
        self.classifier = self.load_model()

    def load_model(self):
        """
        Loads the model from the specified directory, or downloads it if not present.

        Returns:
            transformers.Pipeline: The zero-shot classification pipeline.
        """
        model_dir = os.path.join(self.save_dir, 'model')
        if os.path.exists(model_dir):
            try:
                model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
            except OSError:
                model = TFAutoModelForSequenceClassification.from_pretrained(self.model_name, from_tf=True)
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            try:
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except OSError:
                model = TFAutoModelForSequenceClassification.from_pretrained(self.model_name, from_tf=True)
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
        return pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    def classify(self, text, candidate_labels):
        """
        Classifies the given text against a set of candidate labels.
        """
        # Make sure to truncate the text before classifying
        truncated_text = self.truncate_text(text, self.classifier.tokenizer, max_length=512)
        return self.classifier(truncated_text, candidate_labels)

    @staticmethod
    def truncate_text(text, tokenizer: PreTrainedTokenizerBase, max_length=512):
        """
        Truncates the text to ensure it's within the tokenizer's max length.

        Args:
            text (str): The text to be truncated.
            tokenizer (PreTrainedTokenizerBase): The tokenizer used for the model.
            max_length (int): The maximum length of tokens.

        Returns:
            str: The truncated text.
        """
        # Encode the text and truncate it to the max_length
        inputs = tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=max_length, 
            truncation=True,
            return_overflowing_tokens=False
        )
        # Decode back to a string
        truncated_text = tokenizer.decode(inputs["input_ids"])
        return truncated_text

def save_model(classifier_instance, model_name):
    """
    Saves the model to the specified directory.

    Args:
    classifier_instance (ZeroShotClassifier): The classifier instance to save.
    model_name (str): The name to save the model under.
    """
    model_dir = os.path.join(classifier_instance.save_dir, 'model')
    bentoml.transformers.save_model(model_name, classifier_instance.classifier.model, classifier_instance.classifier.tokenizer, model_dir=model_dir)

def load_model(model_name, save_dir='models'):
    """
    Loads the model with the given name.

    Args:
    model_name (str): The name of the model to load.

    Returns:
    transformers.Pipeline: The loaded zero-shot classification pipeline.
    """
    model_dir = os.path.join(save_dir, model_name.replace('/', '_'), 'model')
    classifier = bentoml.transformers.load_model(model_name, model_dir=model_dir)
    return classifier

if __name__ == "__main__":
    zero_shot_classifier = ZeroShotClassifier()

    save_model(zero_shot_classifier, "my_zero_shot_classifier")

    loaded_classifier = load_model("my_zero_shot_classifier")
    print("Model loaded successfully.")
