"""Optimization demo using saved surrogate models.

Falls back to random search and to a lightweight dummy surrogate if heavy
dependencies (scikit-optimize, xgboost) are not available. Designed as a
portable demo for the project.
"""

import argparse
import os
import numpy as np
import joblib

# Optional import for Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except Exception:
    SKOPT_AVAILABLE = False
    Real = None


def load_model(path):
    """Attempt to load a model with joblib; return a dummy surrogate on error.

    Dummy surrogate returns -0.1 * sum(x) so "lower is better" for demo.
    """
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Warning: failed to load model '{path}': {e}. Using dummy surrogate.")

        class Dummy:
            def __init__(self, n_features=3):
                self.n_features_in_ = n_features

            def predict(self, X):
                X = np.asarray(X)
                return -0.1 * X.sum(axis=1)

        return Dummy()


def objective(x, model):
    x_arr = np.array(x).reshape(1, -1)
    return float(model.predict(x_arr)[0])


def run_demo(model_path='models/model_s11_db.joblib', aux_path='models/model_axial_ratio_db.joblib', n_calls=25):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Primary model not found: {model_path}")

    model = load_model(model_path)

    try:
        n_dim = int(getattr(model, 'n_features_in_'))
    except Exception:
        n_dim = 3

    bounds = [(0.2, 5.0) for _ in range(n_dim)]

    aux_model = None
    if os.path.exists(aux_path):
        aux_model = load_model(aux_path)
        if aux_model is not None:
            print('Loaded auxiliary surrogate (or dummy):', aux_path)

    def wrapped(x):
        val = objective(x, model)
        if aux_model is not None:
            val2 = objective(x, aux_model)
            return float(val + 0.5 * val2)
        return val

    if SKOPT_AVAILABLE:
        sk_bounds = [Real(l, h, name=f'x{i}') for i, (l, h) in enumerate(bounds)]
        print(f'Running gp_minimize with {n_dim} dims, {n_calls} calls')
        res = gp_minimize(wrapped, dimensions=sk_bounds, n_calls=n_calls, random_state=0)
        best_x, best_val = res.x, res.fun
    else:
        print('scikit-optimize not available — running random-search fallback')
        best_x, best_val = None, float('inf')
        for _ in range(n_calls):
            x = [np.random.uniform(l, h) for (l, h) in bounds]
            v = wrapped(x)
            if v < best_val:
                best_val = v
                best_x = x

    print('Best found:', best_x)
    print('Objective:', best_val)
    print('Primary surrogate prediction at best:', float(model.predict(np.array(best_x).reshape(1, -1))[0]))
    if aux_model is not None:
        print('Aux surrogate prediction at best:', float(aux_model.predict(np.array(best_x).reshape(1, -1))[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/model_s11_db.joblib')
    parser.add_argument('--aux', default='models/model_axial_ratio_db.joblib')
    parser.add_argument('--calls', type=int, default=25)
    args = parser.parse_args()
    run_demo(args.model, args.aux, args.calls)


if __name__ == '__main__':
    main()
"""Simple optimization driver using a surrogate model.

This script demonstrates running Bayesian optimization using scikit-optimize
(skopt) on a surrogate model (loaded from models/). It assumes the
surrogate predicts a single scalar objective (e.g., -bandwidth so we maximize).

The script falls back to a simple random-search if scikit-optimize is not
installed so it can run in minimal environments.
"""

import numpy as np
import joblib
import argparse
import os

# Optional skopt import; fallback to random search if unavailable
try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except Exception:
    SKOPT_AVAILABLE = False
    Real = None


def load_model(path):
    try:
        return joblib.load(path)
    except ModuleNotFoundError as e:
        print(f"Warning: could not import module required to load model: {e}. Using dummy surrogate.")

        class DummyModel:
            def __init__(self, n_features=3):
                self.n_features_in_ = n_features

            def predict(self, X):
                X = np.asarray(X)
                vals = -0.1 * X.sum(axis=1)
                return vals

        return DummyModel()
    except Exception as e:
        print(f"Warning: failed to load model '{path}': {e}. Using dummy surrogate.")

        class DummyModel2:
            def __init__(self, n_features=3):
                self.n_features_in_ = n_features

            def predict(self, X):
                X = np.asarray(X)
                vals = -0.1 * X.sum(axis=1)
                return vals

        return DummyModel2()


def objective(x, model):
    x_arr = np.array(x).reshape(1, -1)
    return float(model.predict(x_arr)[0])


def run_example():
    parser = argparse.ArgumentParser(description='Run a short optimization using saved surrogates')
    parser.add_argument('--model', default='models/model_s11_db.joblib', help='Primary surrogate model path')
    parser.add_argument('--aux', default='models/model_axial_ratio_db.joblib', help='Auxiliary surrogate (optional)')
    parser.add_argument('--calls', type=int, default=25, help='Number of optimizer calls')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Primary model not found: {args.model}")

    model = load_model(args.model)

    # Try to infer dimensionality from the model; fall back to 3
    try:
        n_dim = int(getattr(model, 'n_features_in_'))
    except Exception:
        try:
            n_dim = int(model.n_features_in_)
        except Exception:
            n_dim = 3

    # Simple uniform bounds for each parameter — replace with real parameter ranges
    numeric_bounds = [(0.2, 5.0) for _ in range(n_dim)]
    if SKOPT_AVAILABLE:
        bounds = [Real(l, h, name=f'x{i}') for i, (l, h) in enumerate(numeric_bounds)]
    else:
        bounds = numeric_bounds

    aux_model = None
    if os.path.exists(args.aux):
        try:
            aux_model = load_model(args.aux)
            print('Loaded auxiliary model:', args.aux)
        except Exception:
            aux_model = None

    def wrapped(x):
        val = objective(x, model)
        # If we have an auxiliary model (e.g., axial ratio), combine with a simple weight
        if aux_model is not None:
            val2 = objective(x, aux_model)
            # Combine objectives: here we weight axial ratio less (tune as needed)
            return float(val + 0.5 * val2)
        return val

    if SKOPT_AVAILABLE:
        print(f'Running gp_minimize with {n_dim} parameters and {args.calls} calls...')
        res = gp_minimize(wrapped, dimensions=bounds, n_calls=args.calls, random_state=0)
        print('Best found:', res.x)
        print('Objective value:', res.fun)
    else:
        print('scikit-optimize not available — running random search fallback')
        best_x = None
        best_val = float('inf')
        for _ in range(args.calls):
            x = [np.random.uniform(l, h) for (l, h) in numeric_bounds]
            v = wrapped(x)
            if v < best_val:
                best_val = v
                best_x = x
        res = type('R', (), {})()
        res.x = best_x
        res.fun = best_val
        print('Best found (random):', res.x)
        print('Objective value (random):', res.fun)

    # Show surrogate predictions for the best point
    best_pred = model.predict(np.array(res.x).reshape(1, -1))[0]
    print('Primary surrogate prediction at best:', float(best_pred))
    if aux_model is not None:
        print('Aux surrogate prediction at best:', float(aux_model.predict(np.array(res.x).reshape(1, -1))[0]))


if __name__ == '__main__':
    run_example()
