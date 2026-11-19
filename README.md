# Consumer Complaint Response Classification | NLP + XGBoost

**Applied machine learning portfolio project | Antoine Khati, MD, MS**

This repository documents a collaborative University of Connecticut graduate data science project and provides a **separate, reproducible Python companion demonstration**. The original team explored whether consumer complaint narratives and structured metadata could predict company response categories. My contributions included data cleaning, feature engineering, NLP, TPOT AutoML, XGBoost model implementation, and evaluation.

## Original course project: verified results

| Detail | Course project |
| --- | --- |
| Dataset | 100,001 records from the public CFPB Consumer Complaint Database |
| Partitions | 80,000 training observations; 20,001 test observations |
| Target | Six observed company-response classes |
| Methods | Text preprocessing, TF-IDF, structured features, TPOT AutoML, XGBoost, interpretability analysis |
| Held-out weighted F1 | **0.8010** |
| Held-out macro F1 | **0.38** |

The reported result is from the **original collaborative coursework**. Weighted F1 was much higher than macro F1, indicating uneven performance across classes. The original project also included administrative features, such as public response and timing, that may only become available **after** a complaint is submitted or processed. Such features would need to be excluded for an intake-time prediction use case.

## What this repository reproduces

`src/train.py` is a **new educational companion**, built to demonstrate a safer and more transparent workflow using only intake-time predictors:

- TF-IDF on complaint narratives and one-hot encoding of product, issue, and state, all fitted on training data within a scikit-learn `Pipeline`
- XGBoost multiclass classification, stratified holdout evaluation, and a majority-class baseline
- Weighted and macro F1, class-level precision and recall
- Optional training-only randomized hyperparameter search with stratified cross-validation

**The Python demo does not reproduce the original 0.8010 F1 score.** Its synthetic records are deliberately simple, and results from them are only smoke tests. The coursework findings and companion-code results should not be compared as if they came from the same experiment.

## Run

Python 3.10+:

```bash
python -m pip install -r requirements.txt
python src/train.py --demo
python src/train.py --demo --tune
python -m pytest -q
```

For an exploratory run on public CFPB data, download CSV records from the [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) and run:

```bash
python src/train.py --csv /path/to/complaints.csv
```

Required export fields: `Consumer complaint narrative`, `Product`, `Issue`, `State`, and `Company response to consumer`. Free-text narratives may contain sensitive personal information. Review the data source's terms and privacy implications before processing or sharing records. No original complaint dataset, graded notebook, or course materials are published here.

## Files

| Path | Purpose |
| --- | --- |
| `src/train.py` | Preprocessing, model fitting, optional tuning, and evaluation |
| `tests/test_smoke.py` | Synthetic-data smoke tests |
| `requirements.txt` | Dependencies |
| `data/README.md` | Public data source and data-handling notes |

**Limitations:** This is an educational example, not a validated operational model. The target distribution is imbalanced; model selection requires class-sensitive evaluation, careful definition of prediction timing, dataset-shift monitoring, and independent validation.

**Attribution:** Original course project completed collaboratively with Eseoghene Obaruvwe and Ramya Pathman at UConn. Original metrics come from the group analysis. This repository's demonstration source is a separate implementation.
