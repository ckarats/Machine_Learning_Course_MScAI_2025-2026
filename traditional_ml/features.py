"""
features.py
------------
TF-IDF feature extraction with FeatureUnion (word + character n-grams).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


class FeatureExtractor:
    """Dual TF-IDF feature extraction: word-level + character-level.

    Combines two TF-IDF vectorizers via FeatureUnion:
      - Word TF-IDF: unigrams + bigrams
      - Char TF-IDF: character n-grams (3-5) within word boundaries

    Also builds a separate word-only vectorizer for classifiers that
    require non-negative input (e.g. MultinomialNB).

    Parameters
    ----------
    max_word_features : int
        Maximum number of word-level features (default: 10000).
    max_char_features : int
        Maximum number of char-level features (default: 10000).
    word_ngram_range : tuple
        N-gram range for word vectorizer (default: (1, 2)).
    char_ngram_range : tuple
        N-gram range for char vectorizer (default: (3, 5)).
    min_df : int
        Minimum document frequency (default: 2).
    max_df : float
        Maximum document frequency (default: 0.95).
    sublinear_tf : bool
        Apply sublinear TF scaling 1+log(tf) (default: True).
    """

    def __init__(self, max_word_features=10000, max_char_features=10000,
                 word_ngram_range=(1, 2), char_ngram_range=(3, 5),
                 min_df=2, max_df=0.95, sublinear_tf=True):

        self.word_tfidf = TfidfVectorizer(
            max_features=max_word_features,
            ngram_range=word_ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_df=max_df,
            strip_accents='unicode'
        )

        self.char_tfidf = TfidfVectorizer(
            max_features=max_char_features,
            analyzer='char_wb',
            ngram_range=char_ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_df=max_df
        )

        self.feature_union = FeatureUnion([
            ('word', self.word_tfidf),
            ('char', self.char_tfidf)
        ])

        # Separate word-only vectorizer for MultinomialNB (non-negative input)
        self.nb_tfidf = TfidfVectorizer(
            max_features=max_word_features,
            ngram_range=word_ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_df=max_df
        )

        self._fitted = False

    def fit_transform(self, train_texts):
        """Fit on training texts and return transformed matrices.

        Returns
        -------
        X_full : sparse matrix
            FeatureUnion (word + char) matrix.
        X_nb : sparse matrix
            Word-only matrix for MultinomialNB.
        """
        X_full = self.feature_union.fit_transform(train_texts)
        X_nb = self.nb_tfidf.fit_transform(train_texts)
        self._fitted = True

        n_word = len(self.word_tfidf.get_feature_names_out())
        n_char = len(self.char_tfidf.get_feature_names_out())
        print(f"  Features: {X_full.shape[1]} total "
              f"(word: {n_word}, char: {n_char})")

        return X_full, X_nb

    def transform(self, texts):
        """Transform new texts using fitted vectorizers.

        Returns
        -------
        X_full : sparse matrix
        X_nb : sparse matrix
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform() on training data first.")

        X_full = self.feature_union.transform(texts)
        X_nb = self.nb_tfidf.transform(texts)
        return X_full, X_nb

    def get_feature_counts(self):
        """Return dict with feature counts per vectorizer."""
        return {
            'word': len(self.word_tfidf.get_feature_names_out()),
            'char': len(self.char_tfidf.get_feature_names_out()),
            'total': (len(self.word_tfidf.get_feature_names_out())
                      + len(self.char_tfidf.get_feature_names_out())),
            'nb': len(self.nb_tfidf.get_feature_names_out())
        }
