import csv
import os
from datetime import datetime
from math import exp
from random import shuffle, seed


def parse_date(s: str) -> datetime:
    try:
        return datetime.strptime(s.split('.')[0], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime(1970, 1, 1)


def load_demographics(path: str):
    mapping_lists = {
        "bank_account_type": {},
        "bank_name_clients": {},
        "bank_branch_clients": {},
        "employment_status_clients": {},
        "level_of_education_clients": {},
    }
    next_ids = {k: 0 for k in mapping_lists}
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["customerid"]
            birth = datetime.strptime(row["birthdate"].split(" ")[0], "%Y-%m-%d")
            age = 2018 - birth.year
            feats = [age]
            for key in [
                "bank_account_type",
                "bank_name_clients",
                "bank_branch_clients",
                "employment_status_clients",
                "level_of_education_clients",
            ]:
                val = row[key] or ""
                if val not in mapping_lists[key]:
                    mapping_lists[key][val] = next_ids[key]
                    next_ids[key] += 1
                feats.append(mapping_lists[key][val])
            feats.extend(
                [
                    float(row["longitude_gps"] or 0.0),
                    float(row["latitude_gps"] or 0.0),
                ]
            )
            data[cid] = feats
    return data, mapping_lists


def load_prev_loans(path: str):
    stats = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["customerid"]
            stat = stats.setdefault(
                cid,
                {
                    "count": 0,
                    "sum_amount": 0.0,
                    "sum_totaldue": 0.0,
                    "max_loannumber": 0,
                },
            )
            stat["count"] += 1
            stat["sum_amount"] += float(row["loanamount"] or 0.0)
            stat["sum_totaldue"] += float(row["totaldue"] or 0.0)
            loannum = int(row["loannumber"] or 0)
            if loannum > stat["max_loannumber"]:
                stat["max_loannumber"] = loannum
    return stats


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = exp(z)
        return ez / (1.0 + ez)


def train_logreg(X, y, lr=0.01, epochs=500):
    weights = [0.0] * (len(X[0]) + 1)  # bias + weights
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            z = weights[0] + sum(w * x for w, x in zip(weights[1:], xi))
            p = sigmoid(z)
            err = yi - p
            weights[0] += lr * err
            for j, xij in enumerate(xi):
                weights[j + 1] += lr * err * xij
    return weights


def predict(weights, X):
    preds = []
    for xi in X:
        z = weights[0] + sum(w * x for w, x in zip(weights[1:], xi))
        preds.append(1 if z >= 0 else 0)
    return preds


def build_features(row, demo_feats, prev_stats):
    cid = row["customerid"]
    feats = []
    if cid in demo_feats:
        feats.extend(demo_feats[cid])
    else:
        feats.extend([0.0] * 8)  # age + 5 categ enc + 2 coords
    loannum = int(row["loannumber"] or 0)
    loanamount = float(row["loanamount"] or 0.0)
    totaldue = float(row["totaldue"] or 0.0)
    termdays = int(row["termdays"] or 0)
    ratio = totaldue / loanamount if loanamount else 0.0
    creation = parse_date(row["creationdate"])
    approved = parse_date(row["approveddate"])
    days_to_approve = (approved - creation).days
    referred = 0 if not row["referredby"] else 1
    feats.extend([
        loannum,
        loanamount,
        totaldue,
        termdays,
        ratio,
        days_to_approve,
        referred,
    ])
    stat = prev_stats.get(cid)
    if stat:
        avg_prev_amount = stat["sum_amount"] / stat["count"]
        avg_prev_due = stat["sum_totaldue"] / stat["count"]
        feats.extend([stat["count"], avg_prev_amount, avg_prev_due, stat["max_loannumber"]])
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])
    return feats


def load_data():
    demo, _ = load_demographics("data/traindemographics.csv")
    if not os.path.exists("data/trainprevloans.csv"):
        import zipfile

        with zipfile.ZipFile("data/trainprevloans.zip") as zf:
            zf.extractall("data")
    prev = load_prev_loans("data/trainprevloans.csv")
    X, y = [], []
    with open("data/trainperf.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append(build_features(row, demo, prev))
            y.append(1 if row["good_bad_flag"].lower() == "good" else 0)
    # scale features
    mins = [min(col) for col in zip(*X)]
    maxs = [max(col) for col in zip(*X)]
    ranges = [mx - mn if mx - mn else 1 for mn, mx in zip(mins, maxs)]
    for i in range(len(X)):
        X[i] = [(X[i][j] - mins[j]) / ranges[j] for j in range(len(X[i]))]
    return X, y, demo, prev, mins, ranges


def load_test_data(demo_feats, prev_stats, mins, ranges):
    X = []
    ids = []
    demo_test, _ = load_demographics("data/testdemographics.csv")
    if not os.path.exists("data/testprevloans.csv"):
        import zipfile

        with zipfile.ZipFile("data/testprevloans.zip") as zf:
            zf.extractall("data")
    with open("data/testperf.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["customerid"]
            feats = build_features(row, {**demo_feats, **demo_test}, prev_stats)
            feats = [(feats[j] - mins[j]) / ranges[j] for j in range(len(feats))]
            X.append(feats)
            ids.append(cid)
    return ids, X


def cross_val_score(X, y, k=5, epochs=500):
    indices = list(range(len(X)))
    seed(0)
    shuffle(indices)
    fold_size = len(X) // k
    scores = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size
        test_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        X_train = [X[j] for j in train_idx]
        y_train = [y[j] for j in train_idx]
        X_test = [X[j] for j in test_idx]
        y_test = [y[j] for j in test_idx]
        w = train_logreg(X_train, y_train, epochs=epochs)
        preds = predict(w, X_test)
        acc = sum(p == t for p, t in zip(preds, y_test)) / len(y_test)
        scores.append(acc)
    return sum(scores) / len(scores)


def main():
    X, y, demo, prev, mins, ranges = load_data()
    score = cross_val_score(X, y)
    print(f"CV accuracy: {score:.4f}")

    weights = train_logreg(X, y)
    ids, X_test = load_test_data(demo, prev, mins, ranges)
    preds = predict(weights, X_test)
    with open("submission.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["customerid", "Good_Bad_flag"])
        for cid, p in zip(ids, preds):
            writer.writerow([cid, p])
    print("Saved predictions to submission.csv")


if __name__ == "__main__":
    main()
