import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
import string

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")


def extract_keywords(text: str, top_n: int = 3) -> list:
    """
    Extract the most frequent nouns from the text
    """
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove punctuation and stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [
        word
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]

    # Get part-of-speech tags
    pos_tags = pos_tag(tokens)

    # Extract nouns (NN, NNS, NNP, NNPS)
    nouns = [word for word, pos in pos_tags if pos.startswith("NN")]

    # Count the frequency of nouns
    noun_freq = Counter(nouns)

    # Get the top N most frequent nouns
    top_nouns = [word for word, freq in noun_freq.most_common(top_n)]

    return top_nouns
