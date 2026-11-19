"""Illustrative intake-time CFPB company-response classifier.

This standalone educational companion does not reproduce the original
graduate course project or redistribute its dataset.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

# Exclude features generated after company response to reduce timing leakage.
FEATURES = ["narrative", "product", "issue", "state"]
LABEL = "company_response"
INPUT_ALIASES = {
    "Consumer complaint narrative": "narrative",
    "Product": "product",
    "Issue": "issue",
    "State": "state",
    "Company response to consumer": LABEL,
    "consumer_complaint_narrative": "narrative",
    "company_response_to_consumer": LABEL,
}


def clean_text(value: str) -> str:
    """Normalize case, punctuation, whitespace, and redaction markers."""
    value = str(value).lower()
    value = re.sub(r"\bx{3,}\b", " ", value)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", value)).strip()


def make_demo_frame() -> pd.DataFrame:
    """Produce a small, fabricated dataset; no real complaint records."""
    templates = [
        ("Closed with explanation", "The bank explained the account fee and statement", "Checking", "Fees"),
        ("Closed with monetary relief", "A credit card refund was issued after a billing error", "Credit card", "Billing"),
        ("Closed with non-monetary relief", "The credit report entry was corrected after review", "Credit reporting", "Report accuracy"),
        ("In progress", "The complaint is pending investigation and awaiting review", "Student loan", "Servicing"),
        ("Untimely response", "Response delayed and company missed the follow up deadline", "Mortgage", "Response delay"),
        ("Closed", "The customer case was closed and resolved administratively", "Vehicle loan", "Administration"),
    ]
    records = []
    for response, narrative, product, issue in templates:
        for i in range(40):
            records.append({
                "narrative": f"{narrative}. Reference {i % 4}.",
                "product": product,
                "issue": issue,
                "state": ["CT", "NY", "MA", "CA"][i % 4],
                LABEL: response,
            })
    return pd.DataFrame(records)


def load_frame(path: Path) -> pd.DataFrame:
    """Load exported CFPB records and map standard CSV column headings."""
    data = pd.read_csv(path, low_memory=False).rename(columns=INPUT_ALIASES)
    missing = sorted(set(FEATURES + [LABEL]) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = data[FEATURES + [LABEL]].replace({"?": np.nan}).copy()
    frame = frame.dropna(subset=[LABEL])
    for field in FEATURES:
        frame[field] = frame[field].fillna("").astype(str)
    return frame


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(
                max_features=2000, ngram_range=(1, 2),
                preprocessor=clean_text, lowercase=False,
            ), "narrative"),
            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             ["product", "issue", "state"]),
        ],
        remainder="drop",
    )
    estimator = XGBClassifier(
        n_estimators=65, learning_rate=0.08, max_depth=4,
        subsample=0.9, colsample_bytree=0.9,
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=42, n_jobs=2,
    )
    return Pipeline([("features", preprocessor), ("model", estimator)])


def run(frame: pd.DataFrame, tune: bool = False) -> dict:
    """Train/evaluate the companion model, distinct from original coursework."""
    frame = frame.copy()
    frame["narrative"] = frame["narrative"].fillna("").astype(str)
    for field in ("product", "issue", "state"):
        frame[field] = frame[field].fillna("Unknown").astype(str)
    counts = frame[LABEL].value_counts()
    if len(counts) < 2 or counts.min() < 3:
        raise ValueError("Need at least two classes with three examples per class.")

    labels = LabelEncoder()
    y = labels.fit_transform(frame[LABEL])
    X = frame[FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    model = build_pipeline()
    if tune:
        if np.bincount(y_train).min() < 3:
            raise ValueError("Need at least three training rows per class for CV.")
        search = RandomizedSearchCV(
            model,
            param_distributions={
                "model__max_depth": [2, 3, 4, 5],
                "model__n_estimators": [30, 50, 80],
                "model__learning_rate": [0.05, 0.08, 0.12],
            },
            n_iter=5,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring="f1_weighted", random_state=42, n_jobs=1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print("Best cross-validation parameters:", search.best_params_)
    else:
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    metrics = {
        "test_weighted_f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "test_macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "majority_weighted_f1": f1_score(
            y_test, majority.predict(X_test), average="weighted", zero_division=0
        ),
        "test_n": len(y_test),
    }
    print("Metrics from this run (NOT the original course-project metrics):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print("\n", classification_report(
        y_test, predictions, labels=np.arange(len(labels.classes_)),
        target_names=labels.classes_, zero_division=0,
    ))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Generate synthetic sample records")
    source.add_argument("--csv", type=Path, help="Path to locally downloaded CFPB CSV")
    parser.add_argument("--tune", action="store_true", help="Small training-only CV search")
    args = parser.parse_args()
    frame = make_demo_frame() if args.demo else load_frame(args.csv)
    if args.demo:
        print("DEMO MODE: synthetic toy data; performance is illustrative only.")
    run(frame, tune=args.tune)


if __name__ == "__main__":
    main()
