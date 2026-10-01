"""
Are tags landing on the right maps? Side-by-side per-tag quality for model arms.

The gate asks one question - is micro F1 at least as good - and that is the
right question for "may this ship". It is the wrong one for "are tags
misplaced": a model can hold micro F1 steady while still putting 1-2 on maps
with no back-and-forth jumps and missing it on maps full of them. This reports
what that complaint is actually about:

  per-tag precision / recall / AUC   is the tag on the maps that have it?
  sibling correlation gap            do related tags (bursts / triples) move
                                     together in the model far more than they
                                     do in the labels? A big gap means the
                                     features cannot tell them apart.
  tags per map                       predicted vs true, for context

An ARM is one or more model directories of the same configuration (e.g. ten
seeds). Metrics are averaged over the arm and its spread is shown, so a
difference only counts if it clears ordinary seed-to-seed noise.

    python -m tools.tag_quality \\
        --arm champion . \\
        --arm v1-58 candidates/labels58-seed-1 candidates/labels58-seed-2 ... \\
        --arm v2 candidates/v2-seed-1 candidates/v2-seed-2 candidates/v2-seed-3

Uses the frozen holdout. Iterate on features with tools/feature_probe.py
instead, which never touches it.
"""
import argparse
import os

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from mlops.labels import THRESHOLD, predicted  # noqa: E402

# The pattern tags the misplaced-tag complaint is about: each names something a
# feature can look for in the hit objects.
PATTERN_TAGS = [
    '1-2', 'jumps', 'short jumps', 'large jumps', 'cross screen jumps', 'high spacing',
    'sharp angles', 'wide angles', 'square jumps', 'linear aim', 'vertical jumps',
    'flow aim', 'snap aim', 'doubles', 'triples', 'bursts', 'streams', 'deathstream',
    'spaced streams', 'variable streams', 'cut streams', 'alternating', 'slider jumps',
    'burst sliders', 'buzz sliders', 'variable bpm',
]

# Pairs the v1 ensemble moved together far more than the labels do (measured
# on the holdout; see the plan's findings).
SIBLING_PAIRS = [
    ('bursts', 'triples'), ('aim diff spike', 'sharp angles'), ('spaced streams', 'variable streams'),
    ('slider tech', 'tech'), ('large sliders', 'slider aim'), ('aim', 'aim diff spike'),
    ('streams', 'variable streams'), ('awkward aim', 'snap aim'), ('spaced streams', 'streams'),
    ('aim consistency', 'high spacing'), ('cut streams', 'streams'), ('jumps', 'short jumps'),
]


def tag_metrics(y_true, probs, classes, threshold=THRESHOLD):
    """Per-tag precision, recall, AUC and prediction count, plus micro totals."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).astype(bool)
    pred = predicted(probs, threshold)
    tp = (pred & y_true).sum(0)
    fp = (pred & ~y_true).sum(0)
    fn = (~pred & y_true).sum(0)
    per_tag = {}
    for i, tag in enumerate(classes):
        pos = y_true[:, i].sum()
        auc = roc_auc_score(y_true[:, i], probs[:, i]) if 0 < pos < len(y_true) else np.nan
        per_tag[tag] = {
            'precision': tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] else np.nan,
            'recall': tp[i] / pos if pos else np.nan,
            'auc': auc, 'predicted': int(pred[:, i].sum()), 'support': int(pos),
        }
    micro_p = tp.sum() / max(tp.sum() + fp.sum(), 1)
    micro_r = tp.sum() / max(tp.sum() + fn.sum(), 1)
    summary = {
        'micro_precision': micro_p, 'micro_recall': micro_r,
        'micro_f1': 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-12),
        'false_positives': int(fp.sum()), 'true_positives': int(tp.sum()),
        'pred_tags_per_map': pred.sum(1).mean(), 'true_tags_per_map': y_true.sum(1).mean(),
    }
    return summary, per_tag


def sibling_gaps(y_true, probs, classes):
    """corr(model probabilities) - corr(labels), per sibling pair present in classes."""
    index = {c: i for i, c in enumerate(classes)}
    gaps = {}
    for a, b in SIBLING_PAIRS:
        if a in index and b in index:
            i, j = index[a], index[b]
            model_corr = np.corrcoef(probs[:, i], probs[:, j])[0, 1]
            label_corr = np.corrcoef(y_true[:, i], y_true[:, j])[0, 1]
            gaps['%s / %s' % (a, b)] = (model_corr, label_corr)
    return gaps


def evaluate_arm(y_true, prob_list, classes, threshold=THRESHOLD):
    """Average tag_metrics over an arm's runs; keep the per-run values for spread."""
    runs = [tag_metrics(y_true, p, classes, threshold) for p in prob_list]
    gaps = [sibling_gaps(y_true, p, classes) for p in prob_list]
    return runs, gaps


def _mean_std(values):
    values = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    return values.mean(), (values.std(ddof=1) if values.size > 1 else 0.0)


def _cell(values, fmt='%.3f'):
    mean, std = _mean_std(values)
    if np.isnan(mean):
        return '-'
    return (fmt % mean) + ('' if not std else ' ±' + (fmt % std).lstrip('0'))


def format_report(arms, classes, tags=PATTERN_TAGS):
    """arms: {name: (runs, gaps)}. Returns printable text."""
    names = list(arms)
    width = 17
    out = []

    def row(label, cells):
        out.append('%-24s' % label + ''.join('%*s' % (width, c) for c in cells))

    row('', names)
    row('runs per arm', [str(len(arms[n][0])) for n in names])
    for key, label in [('micro_precision', 'micro precision'), ('micro_recall', 'micro recall'),
                       ('micro_f1', 'micro F1'), ('false_positives', 'false positives'),
                       ('pred_tags_per_map', 'pred tags / map')]:
        fmt = '%.0f' if key == 'false_positives' else '%.3f'
        row(label, [_cell([s[key] for s, _ in arms[n][0]], fmt) for n in names])
    row('true tags / map', ['%.3f' % arms[n][0][0][0]['true_tags_per_map'] for n in names])

    for metric in ('precision', 'recall', 'auc'):
        out.append('')
        out.append('PER-TAG %s (at threshold %.2f)' % (metric.upper(), THRESHOLD) if metric != 'auc'
                   else 'PER-TAG AUC (threshold-free: can the model rank maps by this tag at all?)')
        for tag in tags:
            if tag in classes:
                row('  ' + tag, [_cell([pt[tag][metric] for _, pt in arms[n][0]]) for n in names])

    out.append('')
    out.append('SIBLING PAIRS: corr(model probs) vs corr(labels)')
    pairs = list(arms[names[0]][1][0])
    for pair in pairs:
        label_corr = arms[names[0]][1][0][pair][1]
        row('  ' + pair[:22], ['%s (lbl %.2f)' % (_cell([g[pair][0] for g in arms[n][1]], '%.2f'),
                                                  label_corr) for n in names])
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--arm', nargs='+', action='append', required=True, metavar=('NAME', 'DIR'),
                    help='An arm name followed by one or more model directories')
    ap.add_argument('--dataset', default='ml_dataset.json')
    ap.add_argument('--threshold', type=float, default=THRESHOLD)
    args = ap.parse_args()

    from mlops import split as split_mod
    from mlops.scoring import score_on_holdout

    arms = {spec[0]: spec[1:] for spec in args.arm}
    versions = {split_mod.model_feature_version(d) for dirs in arms.values() for d in dirs}
    prepared = split_mod.prepare_for_versions(args.dataset, versions)
    splits = {v: split_mod.fixed_split(p) for v, p in prepared.items()}
    if len({s.split_hash for s in splits.values()}) != 1:
        raise SystemExit('Feature versions disagree on the holdout; refusing to compare.')

    any_prepared = next(iter(prepared.values()))
    y_true = any_prepared.y[next(iter(splits.values())).test_idx]
    results = {}
    for name, dirs in arms.items():
        probs = []
        for d in dirs:
            v = split_mod.model_feature_version(d)
            _, _, p = score_on_holdout(d, prepared[v], splits[v], threshold=args.threshold)
            probs.append(p)
        results[name] = evaluate_arm(y_true, probs, any_prepared.classes, args.threshold)

    print()
    print(format_report(results, any_prepared.classes))


if __name__ == '__main__':
    main()
