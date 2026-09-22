"""
Loading a trained ensemble from a directory and scoring it on the fixed split.

Shared by `cli.py evaluate --holdout` and `cli.py promote`, which is the point:
the gate is only meaningful if the candidate and the champion are measured by
the same code on the same rows. Anything that scores a model goes through
score_on_holdout() so there is exactly one such code path.
"""
import os
import pickle

import numpy as np

from ensemble_evaluator import THRESHOLD

MODEL_GLOB = 'ensemble_model_%d.keras'
SCALER_NAME = 'ensemble_scaler.pkl'
BINARIZER_NAME = 'ensemble_binarizer.pkl'


def ensemble_paths(model_dir, num_models=5):
    return [os.path.join(model_dir, MODEL_GLOB % i) for i in range(1, num_models + 1)]


def count_models(model_dir, max_models=10):
    """How many ensemble_model_N.keras files a directory actually holds."""
    n = 0
    while n < max_models and os.path.exists(os.path.join(model_dir, MODEL_GLOB % (n + 1))):
        n += 1
    return n


def load_ensemble(model_dir, num_models=None):
    """
    Load the models, scaler and binarizer that make up one ensemble.

    Each directory carries its OWN scaler, and that matters: a candidate's
    scaler is refit on the dataset, so scoring a candidate through the
    champion's scaler (or vice versa) would silently measure the wrong thing.
    """
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf

    if num_models is None:
        num_models = count_models(model_dir)
    if num_models == 0:
        raise FileNotFoundError("No ensemble_model_*.keras found in %s" % model_dir)

    scaler_path = os.path.join(model_dir, SCALER_NAME)
    binarizer_path = os.path.join(model_dir, BINARIZER_NAME)
    for p in (scaler_path, binarizer_path):
        if not os.path.exists(p):
            raise FileNotFoundError("Missing %s" % p)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(binarizer_path, 'rb') as f:
        binarizer = pickle.load(f)

    models = [tf.keras.models.load_model(p, compile=False)
              for p in ensemble_paths(model_dir, num_models)]
    return models, scaler, binarizer


def ensemble_probabilities(models, scaler, X_unscaled):
    """Average the per-model sigmoid outputs - the ensemble's actual prediction."""
    X_scaled = scaler.transform(X_unscaled)
    preds = [m.predict(X_scaled, verbose=0) for m in models]
    return np.mean(preds, axis=0)


def score_on_holdout(model_dir, prepared, split, threshold=THRESHOLD, num_models=None):
    """
    Score one model directory on the fixed holdout.

    Refuses to score a model whose label space differs from the dataset's. Such
    a comparison looks fine numerically and is meaningless: column 7 would be a
    different tag for each model. Failing loudly here is the whole reason the
    gate can be trusted.
    """
    from mlops.metrics_report import evaluate_probabilities

    models, scaler, binarizer = load_ensemble(model_dir, num_models)

    model_classes = list(binarizer.classes_)
    if model_classes != list(prepared.classes):
        raise ValueError(
            "Label space mismatch: %s has %d labels, dataset has %d. "
            "These models were trained on a different label vocabulary, so their "
            "scores are not comparable. Retrain the candidate on this dataset."
            % (model_dir, len(model_classes), len(prepared.classes)))

    expected_features = getattr(scaler, 'n_features_in_', None)
    if expected_features is not None and expected_features != prepared.X.shape[1]:
        raise ValueError(
            "Feature count mismatch: %s expects %d features, dataset has %d."
            % (model_dir, expected_features, prepared.X.shape[1]))

    X_holdout = prepared.X[split.test_idx]
    y_holdout = prepared.y[split.test_idx]

    probs = ensemble_probabilities(models, scaler, X_holdout)
    summary, per_tag = evaluate_probabilities(
        y_holdout, probs, prepared.classes, threshold=threshold)

    summary['model_dir'] = model_dir
    summary['num_models'] = len(models)
    summary['split_hash'] = split.split_hash
    summary['dataset_sha256'] = prepared.dataset_sha
    return summary, per_tag, probs
