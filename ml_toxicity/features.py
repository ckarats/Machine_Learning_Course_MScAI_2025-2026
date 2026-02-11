# ml_toxicity/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Tuple

import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class VectorizerConfig:
    max_features: int = 10000
    word_ngrams: Tuple[int, int] = (1, 2)
    char_ngrams: Tuple[int, int] = (3, 5)
    stop_words: Optional[Iterable[str] | str] = "english"
    strip_accents: Optional[str] = "unicode"
    sublinear_tf: bool = True


def clean_text_series(s: pd.Series) -> pd.Series:
    """Minimal safe cleaner."""
    return s.fillna("").astype(str).str.strip()


def build_feature_union(cfg: VectorizerConfig = VectorizerConfig()) -> FeatureUnion:
    """
    Word + char TF-IDF FeatureUnion.
    Fit MUST happen only on training data (handled by sklearn Pipeline).
    """
    word_tfidf = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.word_ngrams,
        sublinear_tf=cfg.sublinear_tf,
        strip_accents=cfg.strip_accents,
        stop_words=cfg.stop_words,
    )

    char_tfidf = TfidfVectorizer(
        max_features=cfg.max_features,
        analyzer="char_wb",
        ngram_range=cfg.char_ngrams,
        sublinear_tf=cfg.sublinear_tf,
    )

    return FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])


def build_nb_vectorizer(cfg: VectorizerConfig = VectorizerConfig()) -> TfidfVectorizer:
    """
    A separate TF-IDF for MultinomialNB is often useful (simple, stable).
    """
    return TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.word_ngrams,
        sublinear_tf=cfg.sublinear_tf,
        strip_accents=cfg.strip_accents,
        stop_words=cfg.stop_words,
    )
