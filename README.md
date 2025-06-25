# Loan Default Prediction Challenge

This repository contains a simple baseline solution for the "Loan Default Prediction" competition. The goal is to predict whether a customer will repay a loan. The scoring metric is the percentage of incorrect answers (1 - accuracy).

## Baseline model

`baseline.py` implements a logistic regression trained only with the Python standard library. The script automatically extracts the zipped data files on first run, builds numeric features from all available tables and performs 5-fold cross validation. It then trains on the full training set and saves predictions for the test set into `submission.csv`.

Cross‑validation accuracy is around **0.78**, which corresponds to an error rate below **0.22**, slightly better than the trivial approach of predicting all ones.

## Usage

```bash
python3 baseline.py
```

After running, check `submission.csv` for the predictions.
