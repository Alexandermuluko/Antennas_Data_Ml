"""Train surrogate models from a CSV dataset.

Usage example:
    python scripts/train_surrogate_fixed.py data/dataset.csv --params w1,w2,h1 --targets s11_gain,ar

This script trains one XGBoost regressor per target column and saves the
trained models under `models/`.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def train_models(csv_path, params_cols, target_cols, out_dir='models'):
    """Train regressors for the given target columns and save them.

    csv_path: path to a CSV with columns for parameters and targets
    params_cols: list of input parameter column names
    target_cols: list of target column names to predict
    out_dir: directory to store trained models
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    X = df[params_cols].values

    for target in target_cols:
        y = df[target].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(n_estimators=200, max_depth=6)
        model.fit(Xtr, ytr, eval_set=[(Xte, yte)], early_stopping_rounds=20, verbose=False)

        path = os.path.join(out_dir, f'model_{target}.joblib')
        joblib.dump(model, path)
        print('Saved', path)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('csv', help='input dataset CSV')
    p.add_argument('--params', required=True, help='comma-separated parameter column names')
    p.add_argument('--targets', required=True, help='comma-separated target column names')
    p.add_argument('--out', default='models', help='output models directory')
    args = p.parse_args()

    params = args.params.split(',')
    targets = args.targets.split(',')
    train_models(args.csv, params, targets, args.out)
