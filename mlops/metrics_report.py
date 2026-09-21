"""
Turning predicted probabilities into the numbers the gate argues about.

Only one number actually decides a promotion - macro F1 - but macro F1 alone is
a bad thing to look at on its own here. With 66 labels of wildly uneven support,
it weights a tag appearing 12 times exactly as heavily as one appearing 700
times, so it moves for reasons that have nothing to do with the model getting
better at the common cases. That is precisely why it is the gate metric (a model
that quietly abandons rare tags should not pass), and precisely why the per-tag
CSV is logged next to it: the summary says whether to promote, the CSV says why.
"""
import numpy as np

# Kept in sync with ensemble_evaluator.THRESHOLD by importing it rather than
# repeating the literal - a second copy of 0.27 is a second thing to forget.
from ensemble_evaluator import THRESHOLD


def evaluate_probabilities(y_true, probs, classes, threshold=THRESHOLD):
    """
    Score averaged ensemble probabilities against binary truth.

    Scores RAW thresholded output. The predict path in ensemble_evaluator and
    neural_model additionally applies an expert-system override that forces the
    'streams' tag on maps with a 15+ note sequence; that override is deliberately
    NOT applied here, matching the existing classification_report and keeping the
    gate measuring the network rather than the network plus a hand-written rule.
    So these numbers describe the model, not end-to-end app behaviour.

    Returns (summary dict, per-tag pandas DataFrame).
    """
    import pandas as pd
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    if y_true.shape != probs.shape:
        raise ValueError(
            "truth/prediction shape mismatch: %s vs %s" % (y_true.shape, probs.shape))
    if len(classes) != y_true.shape[1]:
        raise ValueError(
            "got %d class names for %d label columns" % (len(classes), y_true.shape[1]))

    y_pred = (probs >= threshold).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)

    per_tag = pd.DataFrame({
        'tag': list(classes),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'predicted': y_pred.sum(axis=0),
    }).sort_values('f1', ascending=False).reset_index(drop=True)

    summary = {
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'samples_f1': float(f1_score(y_true, y_pred, average='samples', zero_division=0)),
        'threshold': float(threshold),
        'n_samples': int(y_true.shape[0]),
        'n_labels': int(y_true.shape[1]),
        # How many of the 66 tags the model never predicts at all. A model can
        # lift micro F1 while silently going mute on rare tags, and that shows
        # up here before it shows up anywhere else.
        'labels_never_predicted': int((y_pred.sum(axis=0) == 0).sum()),
    }
    return summary, per_tag


def format_summary(summary):
    """One-line human-readable form, for terminal output and commit-free logs."""
    return (
        "macro_f1=%.4f micro_f1=%.4f weighted_f1=%.4f "
        "(threshold=%.2f, n=%d, %d/%d tags never predicted)" % (
            summary['macro_f1'], summary['micro_f1'], summary['weighted_f1'],
            summary['threshold'], summary['n_samples'],
            summary['labels_never_predicted'], summary['n_labels']))
