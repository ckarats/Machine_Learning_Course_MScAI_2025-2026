"""
traditional_ml
===============
Modular pipeline for text classification using traditional ML methods.

Modules
-------
- preprocessing : Tweet text cleaning and normalization
- features      : TF-IDF feature extraction (word + char n-grams)
- models        : Model definitions, GridSearchCV, SMOTE, evaluation
- pipeline      : High-level binary and multi-class orchestration
"""

from .preprocessing import TweetPreprocessor
from .features import FeatureExtractor
from .models import (apply_smote, run_gridsearch, evaluate_model,
                     print_summary, get_base_models,
                     build_voting_ensemble, PARAM_GRIDS)
from .pipeline import run_binary_classification, run_multiclass_classification

__all__ = [
    'TweetPreprocessor',
    'FeatureExtractor',
    'apply_smote',
    'run_gridsearch',
    'evaluate_model',
    'print_summary',
    'get_base_models',
    'build_voting_ensemble',
    'run_binary_classification',
    'run_multiclass_classification',
    'PARAM_GRIDS',
]
