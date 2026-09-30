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

# Minimum holdout support for a tag's per-tag F1 to be treated as a measurement
# rather than a coin flip. One map moves a tag's recall by 1/support, so at
# support 10 a single flip is ~10% of the score and below that it dominates.
#
# Chosen from that property of the holdout, NOT by picking whichever floor made
# the gate look best - support >= 30 gives a lower spread and was not chosen.
# On this holdout the floor keeps 49 of 66 tags. See VERIFIED.md.
SUPPORT_FLOOR = 10


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

    # Split the tags by whether their support is large enough for a per-tag F1
    # to mean anything. One holdout map moves a tag's recall by 1/support, so
    # below SUPPORT_FLOOR a single flip dominates the score. See SUPPORT_FLOOR.
    included = support >= SUPPORT_FLOOR
    never = y_pred.sum(axis=0) == 0

    per_tag = pd.DataFrame({
        'tag': list(classes),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'predicted': y_pred.sum(axis=0),
    }).sort_values('f1', ascending=False).reset_index(drop=True)

    summary = {
        # The gate metric. Micro pools predictions instead of averaging per-tag
        # scores, which makes it ~6x steadier run-to-run than macro here.
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'samples_f1': float(f1_score(y_true, y_pred, average='samples', zero_division=0)),
        # Logged for diagnosis, not gated on: it changed no verdict in the
        # validation set, and an inert rule is just another rule to explain.
        'macro_f1_supported': float(f1[included].mean()) if included.any() else 0.0,
        'threshold': float(threshold),
        'support_floor': int(SUPPORT_FLOOR),
        'n_samples': int(y_true.shape[0]),
        'n_labels': int(y_true.shape[1]),
        'n_labels_supported': int(included.sum()),
        # How many tags the model never predicts at all. A model can lift micro
        # F1 while going mute on rarer tags, and that shows up here first.
        #
        # The gate uses the SUPPORTED count only. The full count swings 11..16
        # across identical reruns (sigma 1.34), so a rule on it rejects honest
        # models; restricted to tags with real support it is 1..2 (sigma 0.42).
        'labels_never_predicted': int(never.sum()),
        'labels_never_predicted_supported': int((never & included).sum()),
        'labels_never_predicted_rare': int((never & ~included).sum()),
    }
    return summary, per_tag


def format_summary(summary):
    """
    One-line human-readable form. micro_f1 comes first because it is the gate
    metric; macro is shown alongside it but is not what decides a promotion.
    """
    return (
        "micro_f1=%.4f macro_f1=%.4f weighted_f1=%.4f "
        "(threshold=%.2f, n=%d, never-predicted %d/%d supported, %d/%d overall)" % (
            summary['micro_f1'], summary['macro_f1'], summary['weighted_f1'],
            summary['threshold'], summary['n_samples'],
            summary.get('labels_never_predicted_supported', -1),
            summary.get('n_labels_supported', -1),
            summary['labels_never_predicted'], summary['n_labels']))
