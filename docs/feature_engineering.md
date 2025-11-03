# Feature Engineering Guide

This document describes the comprehensive feature engineering pipeline for the BT4012 text classification competition.

## Overview

We engineer **~100+ features** from raw text data across 6 categories:

1. **Statistical Features** (26 features)
2. **Linguistic Features** (24 features)
3. **Sentiment Features** (11 features)
4. **N-gram Features** (8 features)
5. **Domain-Specific Features** (21 features)
6. **TF-IDF Features** (500 features)

## Feature Categories

### 1. Statistical Features

Basic text statistics and character-level features:

- **Length metrics**: character count, word count, sentence count
- **Average metrics**: average word length, average sentence length
- **Punctuation counts**: commas, semicolons, periods, questions, exclamations, quotes, parentheses, dashes
- **Punctuation ratio**: total punctuation / character count
- **Capitalization**: uppercase count, uppercase ratio, title word count
- **Digits**: digit count, digit ratio
- **Whitespace**: space count, newline count
- **Special characters**: count of non-alphanumeric, non-space characters

### 2. Linguistic Features

Advanced language analysis:

- **Vocabulary richness**:
  - Unique word count
  - Lexical diversity (unique words / total words)
- **Stopword analysis**:
  - Stopword count
  - Stopword ratio
- **Word length statistics**:
  - Min/max/std of word lengths
  - Short word count (≤ 3 chars)
  - Long word count (≥ 10 chars)
- **Readability scores**:
  - Flesch Reading Ease
  - Flesch-Kincaid Grade
  - Gunning Fog Index
  - SMOG Index
  - Automated Readability Index
  - Coleman-Liau Index
  - Syllable count
  - Polysyllable count
- **Part-of-Speech ratios**:
  - Noun ratio
  - Verb ratio
  - Adjective ratio
  - Adverb ratio

### 3. Sentiment Features

Sentiment and emotion analysis:

- **Overall sentiment**:
  - Polarity (-1 to 1)
  - Subjectivity (0 to 1)
- **Sentence-level statistics**:
  - Mean polarity
  - Std deviation of polarity
  - Min/max polarity
- **Sentiment distribution**:
  - Positive sentence count
  - Negative sentence count
  - Neutral sentence count

### 4. N-gram Features

Phrase and pattern detection:

- **Bigrams** (2-word sequences):
  - Count
  - Unique count
  - Diversity (unique / total)
  - Max frequency (most repeated bigram)
- **Trigrams** (3-word sequences):
  - Count
  - Unique count
  - Diversity
  - Max frequency

### 5. Domain-Specific Features

Features specific to academic/technical writing:

- **Terminology counts**:
  - Technical terms (model, data, algorithm, etc.)
  - Statistical terms (mean, correlation, distribution, etc.)
  - Methodology terms (approach, technique, experiment, etc.)
  - Evaluation terms (metric, performance, effectiveness, etc.)
- **Quantitative indicators**:
  - Number count
  - Percentage count
  - Decimal count
- **Academic markers**:
  - Citation count (year in parentheses)
  - "et al" count
  - Figure/table/appendix references
- **Writing style**:
  - First-person pronoun count (we, our, I, etc.)
  - Formal conjunction count (however, therefore, etc.)
  - Hedging word count (may, might, possibly, etc.)

### 6. TF-IDF Features

Text representation using Term Frequency-Inverse Document Frequency:

- **Configuration**:
  - Max features: 500
  - N-gram range: (1, 2) - unigrams and bigrams
  - Min document frequency: 5
  - Max document frequency: 0.8
  - Stop words: English
- **Output**: 500 numerical features representing the most important terms/phrases

## Usage

### From Command Line

```bash
# Generate features from training data
python -m bt4012_competition_2.features \
    --input-path data/raw/train.csv \
    --output-path data/processed/train_features.csv
```

### From Python Code

```python
from bt4012_competition_2.features import extract_all_features, ensure_nltk_data
import pandas as pd
import pickle

# Ensure NLTK data is downloaded
ensure_nltk_data()

# Load data
df = pd.read_csv('data/raw/train.csv')

# Extract features for training data
df_features, tfidf_vectorizer = extract_all_features(df, fit_tfidf=True)

# Save vectorizer for test data
with open('data/processed/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# For test data (reuse the fitted vectorizer)
df_test = pd.read_csv('data/raw/test.csv')
df_test_features = extract_all_features(
    df_test,
    tfidf_vectorizer=tfidf_vectorizer,
    fit_tfidf=False
)
```

### From Jupyter Notebook

See `notebooks/02_feature_engineering.ipynb` for a detailed walkthrough with visualizations.

## Output Files

The feature engineering pipeline generates:

1. **train_features.csv**: All engineered features + label column
2. **tfidf_vectorizer.pkl**: Fitted TF-IDF vectorizer (for test data processing)
3. **feature_names.txt**: List of all feature names

## Feature Selection Considerations

With 100+ features, you may want to:

1. **Remove highly correlated features** (correlation > 0.9)
2. **Use feature importance** from tree-based models
3. **Apply dimensionality reduction** (PCA, feature selection)
4. **Try different feature subsets** for ensemble models

## Performance Notes

- Feature extraction takes **2-5 minutes** for ~5,000 samples
- Most time-consuming: linguistic features (POS tagging, readability)
- TF-IDF is fast with sparse matrix representation
- Memory usage: ~50MB for 5,000 samples with all features

## Dependencies

```
pandas
numpy
scikit-learn
nltk
textblob
textstat
```

Install all required packages:

```bash
pip install -r requirements.txt
```
