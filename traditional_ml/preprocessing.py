"""
preprocessing.py
-----------------
Text preprocessing pipeline for tweet classification.
"""

import re
import string

import nltk
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem.snowball import SnowballStemmer


class TweetPreprocessor:
    """Rule-based tweet cleaning pipeline.

    Parameters
    ----------
    language : str
        Language for stopwords and stemmer (default: 'english').
    remove_stopwords : bool
        Whether to remove stopwords. Set to False when using bigrams
        to preserve informative phrases like "kill you", "hate you".
    min_token_len : int
        Minimum token length to keep (default: 2).
    """

    def __init__(self, language='english', remove_stopwords=True,
                 min_token_len=2):
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.min_token_len = min_token_len
        self.stemmer = SnowballStemmer(language)
        self.stopwords = set(nltk_stopwords.words(language))
        self.punct_translator = str.maketrans('', '', string.punctuation)

    def __call__(self, text):
        """Preprocess a single text string."""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)   # URLs
        text = re.sub(r'@\w+', '', text)                       # @mentions
        text = re.sub(r'#(\w+)', r'\1', text)                  # #hashtag → hashtag
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)            # elongated chars
        text = re.sub(r'\d+', 'NUM', text)                     # numbers → NUM
        text = text.translate(self.punct_translator)            # punctuation

        tokens = text.split()

        if self.remove_stopwords:
            tokens = [t for t in tokens
                      if t not in self.stopwords and len(t) >= self.min_token_len]
        else:
            tokens = [t for t in tokens if len(t) >= self.min_token_len]

        tokens = [self.stemmer.stem(t) for t in tokens]
        return " ".join(tokens)

    def transform_series(self, series):
        """Apply preprocessing to a pandas Series."""
        return series.apply(self)
