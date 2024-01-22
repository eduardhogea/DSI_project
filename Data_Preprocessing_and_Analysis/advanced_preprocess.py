"""
Module for advanced preprocessing of text data using NLTK library.
This includes lowercasing, removing non-alphabetic characters, tokenization,
removing stopwords, and lemmatization.
"""

"""
Testing something here for git.

"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def advanced_preprocess_text(text):
    """
    Perform advanced preprocessing on the given text.
    
    This function converts the text to lowercase, removes non-alphabetic characters,
    tokenizes, removes stopwords, and applies lemmatization.

    Parameters:
    text (str): The text to preprocess.

    Returns:
    str: The preprocessed text.
    """
    # Lowercasing and removing non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Removing stopwords and lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

    return ' '.join(filtered_tokens)

if __name__ == "__main__":
    # Test the function with a sample text
    SAMPLE_TEXT = "This is a sample text for testing the advanced preprocessing function."
    print("Original Text:", SAMPLE_TEXT)
    print("Preprocessed Text:", advanced_preprocess_text(SAMPLE_TEXT))
