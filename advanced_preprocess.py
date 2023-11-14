import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def advanced_preprocess_text(text):
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
    sample_text = "This is a sample text for testing the advanced preprocessing function."
    print("Original Text:", sample_text)
    print("Preprocessed Text:", advanced_preprocess_text(sample_text))
