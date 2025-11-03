"""
Training helper for spam-classifier demo.

Usage:
    python train_and_save.py

This script downloads the dataset, trains a small TF-IDF + LogisticRegression
and saves model and vectorizer to the `models/` directory.
"""
from pathlib import Path
import argparse
import joblib
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from text_preprocessing import preprocess_text


TRAIN_DATA_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"


def download_and_load(url_or_path=TRAIN_DATA_URL):
    # Try to read with header=None first (many SMS spam datasets have no header)
    try:
        df = pd.read_csv(url_or_path, header=None, encoding='latin-1')
        # if it has exactly 2 columns assume label/message
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    # Fallback: try with header=0
    df = pd.read_csv(url_or_path, header=0, encoding='utf-8')
    return df


def _get_series(df, col_selector):
    # Support selectors like 'col_0', 'col_1' to reference positional columns
    if col_selector is None:
        return None
    if isinstance(col_selector, str) and col_selector.startswith('col_'):
        try:
            idx = int(col_selector.split('_', 1)[1])
            return df.iloc[:, idx].astype(str)
        except Exception:
            raise ValueError(f"Invalid column selector '{col_selector}' for positional column")
    # Otherwise expect literal column name
    if col_selector in df.columns:
        return df[col_selector].astype(str)
    raise ValueError(f"Column '{col_selector}' not found in dataframe columns: {list(df.columns[:10])}")


def train_and_save(
    data_path=TRAIN_DATA_URL,
    label_col='col_0',
    text_col='col_1',
    out_dir='models',
    sample_size=None,
    test_size=0.2,
    seed=42,
    remove_stopwords=False,
    keep_numbers=False,
    steps_out_dir=None,
    save_step_columns=False,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = download_and_load(data_path)
    if sample_size:
        df = df.sample(min(len(df), sample_size), random_state=seed)

    # extract label and text
    y_series = _get_series(df, label_col)
    X_series = _get_series(df, text_col)

    # Apply preprocessing to each text
    processed_texts = []
    processed_columns = {}
    for text in X_series:
        result = preprocess_text(text, remove_stopwords, keep_numbers)
        processed_texts.append(result['text_clean'])
        
        # Collect step columns if requested
        if save_step_columns:
            for step_name, step_text in result.items():
                if step_name not in processed_columns:
                    processed_columns[step_name] = []
                processed_columns[step_name].append(step_text)

    # Save intermediate steps if requested
    if steps_out_dir is not None:
        steps_dir = Path(steps_out_dir)
        steps_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataframe with all steps
        steps_df = pd.DataFrame({
            'label': y_series,
            'text_original': X_series,
            **processed_columns
        })
        steps_df.to_csv(steps_dir / 'preprocessing_steps.csv', index=False)

    # map labels if they are 'spam'/'ham'
    y = y_series.map(lambda v: 1 if str(v).strip().lower() == 'spam' else 0)

    # Configure vectorizer with recommended settings
    vect = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # unigrams + bigrams
        min_df=2,            # minimum document frequency
        sublinear_tf=True,   # sublinear tf scaling
    )
    Xv = vect.fit_transform(processed_texts)

    X_train, X_test, y_train, y_test = train_test_split(Xv, y, test_size=test_size, random_state=seed)

    # Use recommended LogisticRegression settings
    clf = LogisticRegression(
        C=2.0,               # regularization strength
        class_weight='balanced',  # handle class imbalance
        max_iter=1000,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    metrics = {
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'n_train': int(X_train.shape[0]),
        'n_test': int(X_test.shape[0]),
        'settings': {
            'class_weight': 'balanced',
            'C': 2.0,
            'min_df': 2,
            'ngram_range': [1, 2],
            'sublinear_tf': True,
            'remove_stopwords': remove_stopwords,
            'keep_numbers': keep_numbers,
        }
    }

    joblib.dump(clf, out / 'model.joblib')
    joblib.dump(vect, out / 'vectorizer.joblib')
    with open(out / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print('Saved model, vectorizer, metrics to', out)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='models')
    parser.add_argument('--sample-size', type=int, default=3000, help='Limit training size for speed')
    parser.add_argument('--data-path', default=TRAIN_DATA_URL, help='URL or local CSV path')
    parser.add_argument('--label-col', default='col_0', help='Label column name or col_{index}')
    parser.add_argument('--text-col', default='col_1', help='Text column name or col_{index}')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Preprocessing options
    parser.add_argument('--remove-stopwords', action='store_true', help='Remove stopwords')
    parser.add_argument('--keep-numbers', action='store_true', help='Keep numbers instead of replacing with <NUM>')
    parser.add_argument('--steps-out-dir', help='Directory to save preprocessing step results')
    parser.add_argument('--save-step-columns', action='store_true', help='Save intermediate preprocessing steps')
    
    args = parser.parse_args()
    print('Downloading and training...')
    train_and_save(
        data_path=args.data_path,
        label_col=args.label_col,
        text_col=args.text_col,
        out_dir=args.out_dir,
        sample_size=args.sample_size,
        test_size=args.test_size,
        seed=args.seed,
        remove_stopwords=args.remove_stopwords,
        keep_numbers=args.keep_numbers,
        steps_out_dir=args.steps_out_dir,
        save_step_columns=args.save_step_columns,
    )
