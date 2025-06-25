# Loan Default Prediction Challenge

This repository contains a simple baseline solution for the "Loan Default Prediction" competition. The goal is to predict whether a customer will repay a loan. The scoring metric is the percentage of incorrect answers (1 - accuracy).

## Baseline model

`baseline.py` implements a logistic regression trained only with the Python standard library. The script automatically extracts the zipped data files on first run, builds numeric features from all available tables and performs 5-fold cross validation. It then trains on the full training set and saves predictions for the test set into `submission.csv`.

Cross‑validation accuracy is around **0.78**, which corresponds to an error rate below **0.22**, slightly better than the trivial approach of predicting all ones.

## Advanced model

`sklearn_model.py` uses the libraries from `requirements.txt` to build a more powerful pipeline based on scikit‑learn. It performs one‑hot encoding of the categorical fields and fits a logistic regression with built‑in cross validation. In practice this approach yields an accuracy above **0.80** on the same 5‑fold split.

## Usage

```bash
python3 baseline.py
```

After running, check `submission.csv` for the predictions.

## Requirements

Install the dependencies from `requirements.txt` to run more advanced models:

```bash
pip install -r requirements.txt
```
