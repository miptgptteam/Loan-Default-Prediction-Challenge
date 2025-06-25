import os
import zipfile
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def load_prev_loans(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    agg = df.groupby('customerid').agg(
        count=('loanamount', 'size'),
        avg_amount=('loanamount', 'mean'),
        avg_totaldue=('totaldue', 'mean'),
        max_loannumber=('loannumber', 'max'),
    )
    return agg


def prepare_dataset():
    if not os.path.exists('data/trainprevloans.csv'):
        with zipfile.ZipFile('data/trainprevloans.zip') as zf:
            zf.extractall('data')
    if not os.path.exists('data/testprevloans.csv'):
        with zipfile.ZipFile('data/testprevloans.zip') as zf:
            zf.extractall('data')

    demo_train = pd.read_csv('data/traindemographics.csv')
    demo_test = pd.read_csv('data/testdemographics.csv')

    def preprocess_demo(df: pd.DataFrame) -> pd.DataFrame:
        df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')
        df['age'] = 2018 - df['birthdate'].dt.year
        df = df.drop(columns=['birthdate'])
        return df

    demo_train = preprocess_demo(demo_train)
    demo_test = preprocess_demo(demo_test)

    prev_train = load_prev_loans('data/trainprevloans.csv')
    prev_test = load_prev_loans('data/testprevloans.csv')

    perf_train = pd.read_csv('data/trainperf.csv')
    perf_test = pd.read_csv('data/testperf.csv')

    def preprocess_perf(df: pd.DataFrame) -> pd.DataFrame:
        df['approveddate'] = pd.to_datetime(df['approveddate'])
        df['creationdate'] = pd.to_datetime(df['creationdate'])
        df['days_to_approve'] = (df['approveddate'] - df['creationdate']).dt.days
        df['ratio'] = df['totaldue'] / df['loanamount']
        df['referred'] = df['referredby'].notnull().astype(int)
        df = df.drop(columns=['approveddate', 'creationdate', 'referredby'])
        return df

    perf_train = preprocess_perf(perf_train)
    perf_test = preprocess_perf(perf_test)

    train = perf_train.merge(demo_train, on='customerid', how='left')\
                      .merge(prev_train, on='customerid', how='left')
    test = perf_test.merge(demo_test, on='customerid', how='left')\
                    .merge(prev_test, on='customerid', how='left')

    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    y = (train['good_bad_flag'].str.lower() == 'good').astype(int)
    train = train.drop(columns=['good_bad_flag'])

    test_ids = test['customerid']
    train = train.drop(columns=['customerid'])
    test = test.drop(columns=['customerid'])

    return train, y, test, test_ids


def main():
    X_train, y_train, X_test, test_ids = prepare_dataset()

    categorical_cols = [
        'bank_account_type',
        'bank_name_clients',
        'bank_branch_clients',
        'employment_status_clients',
        'level_of_education_clients',
    ]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        [('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
        remainder='passthrough',
    )

    clf = Pipeline([
        ('prep', preprocessor),
        ('logreg', LogisticRegression(max_iter=1000)),
    ])

    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV accuracy: {scores.mean():.4f}")

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    pd.DataFrame({'customerid': test_ids, 'Good_Bad_flag': preds}).to_csv(
        'submission.csv', index=False
    )
    print('Saved predictions to submission.csv')


if __name__ == '__main__':
    main()
