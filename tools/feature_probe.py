"""
Compare feature sets without touching the holdout.

Feature engineering is a loop - add a feature, look, adjust - and every look
at the frozen 929-map holdout spends it: features tuned until the holdout
likes them make the gate's final comparison optimistic, and nothing would show
it. So this carves a fixed DEV split out of the training rows only (the
holdout rows are never loaded into it), trains single ensemble members with
the exact shipped architecture (osu_tagger.training.ensemble.train_member), and reports
per-tag quality with tools/tag_quality.

    python -m tools.feature_probe --sets v1 v2 --seeds 1 2 3
    python -m tools.feature_probe --sets v2 v2-no-jumps v2-no-chains

A set is v1, v2, v1+v2 (a diagnostic: does v1 still add anything v2 lacks?)
or v2-no-<group> (v2 without one of features_v2's groups: tempo, density,
rhythm, chains, stream_shape, jumps, sliders, overlaps, time_profile). The
scaler is fit on the dev-train rows only.

Threshold questions belong here too, for the same reason: a threshold picked on
the holdout is tuned to it. --members 5 averages five members per seed, like the
shipped ensemble, so the probabilities are calibrated the way a real model's
are; --sweep then reports precision / recall / F1 across thresholds.

    python -m tools.feature_probe --sets v2 --members 5 --sweep --sweep-json sweep.json
    python cli.py threshold-sweep          (the same sweep, with those defaults)
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from mlops.labels import THRESHOLD, predicted  # noqa: E402

DEV_SEED = 7           # independent of the frozen evaluation seed (42)
DEV_SIZE = 0.2


def _columns(spec):
    """[(feature version, column indices or None for all)] making up a set."""
    from osu_tagger.features.v2 import FEATURE_GROUPS_V2, FEATURE_NAMES_V2

    if spec in ('v1', 'v2'):
        return [(int(spec[1]), None)]
    if spec == 'v1+v2':
        # Diagnostic only: if this beats v2, v1 still carries information v2 lacks.
        return [(1, None), (2, None)]
    if spec.startswith('v2-no-'):
        group = spec[len('v2-no-'):]
        if group not in FEATURE_GROUPS_V2:
            raise SystemExit('Unknown v2 group %r; known: %s' % (group, ', '.join(FEATURE_GROUPS_V2)))
        dropped = set(FEATURE_GROUPS_V2[group])
        return [(2, [i for i, n in enumerate(FEATURE_NAMES_V2) if n not in dropped])]
    raise SystemExit('Unknown set %r (use v1, v2, v1+v2 or v2-no-<group>)' % spec)


SWEEP_THRESHOLDS = [round(0.10 + 0.01 * i, 2) for i in range(51)]      # 0.10 .. 0.60


def threshold_sweep(y_true, probs):
    """Micro precision / recall / F1 and tags per map at each threshold."""
    y_true = np.asarray(y_true).astype(bool)
    rows = []
    for t in SWEEP_THRESHOLDS:
        pred = predicted(probs, t)
        tp = (pred & y_true).sum()
        fp = (pred & ~y_true).sum()
        fn = (~pred & y_true).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        rows.append({'threshold': t, 'precision': float(p), 'recall': float(r),
                     'f1': float(2 * p * r / max(p + r, 1e-12)),
                     'false_positives': int(fp), 'pred_tags_per_map': float(pred.sum(1).mean()),
                     'tags_ever_predicted': int(pred.any(0).sum())})
    return rows


def add_arguments(ap, sweep=False):
    """
    Shared by `python -m tools.feature_probe` and `cli.py threshold-sweep`. With
    sweep=True the defaults are the threshold question's: the v2 set, 5-member
    ensembles like the shipped one, and the sweep written to reports/.
    """
    ap.add_argument('--sets', nargs='+', default=['v2'] if sweep else ['v1', 'v2'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--dataset', default='ml_dataset.json')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--threshold', type=float, default=THRESHOLD,
                    help='The threshold the report marks as current (default: %(default)s)')
    ap.add_argument('--members', type=int, default=5 if sweep else 1,
                    help='Ensemble members averaged per seed (the shipped ensemble has 5)')
    if not sweep:
        ap.add_argument('--sweep', action='store_true',
                        help='Also report micro precision/recall/F1 across thresholds 0.10-0.60')
    ap.add_argument('--sweep-json', default=os.path.join('reports', 'threshold_sweep.json') if sweep else None,
                    help='Write the sweep (averaged over seeds) to this JSON file')


def run(args):
    """Returns an exit code, like every cli.py subcommand: 0 on success."""
    if not os.path.exists(args.dataset):
        print('Dataset not found: %s' % args.dataset)
        return 1

    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from osu_tagger.training.ensemble import train_member
    from mlops import split as split_mod
    from tools.tag_quality import evaluate_arm, format_report

    specs = {s: _columns(s) for s in args.sets}
    prepared = split_mod.prepare_for_versions(
        args.dataset, {v for parts in specs.values() for v, _ in parts})
    any_prepared = next(iter(prepared.values()))
    split = split_mod.fixed_split(any_prepared)

    # The dev split lives entirely inside the training rows.
    dev_train, dev_val = train_test_split(split.train_idx, test_size=DEV_SIZE,
                                          random_state=DEV_SEED)
    assert not set(dev_val) & set(split.test_idx)
    y = any_prepared.y
    print('dev split: %d train / %d val rows, all from the %d training rows; holdout untouched'
          % (len(dev_train), len(dev_val), len(split.train_idx)))

    results, sweeps = {}, {}
    for spec, parts in specs.items():
        X = np.hstack([prepared[v].X if cols is None else prepared[v].X[:, cols]
                       for v, cols in parts])
        scaler = StandardScaler().fit(X[dev_train])
        X_train, X_val = scaler.transform(X[dev_train]), scaler.transform(X[dev_val])
        probs = []
        for seed in args.seeds:
            tf.keras.utils.set_random_seed(seed)
            members = [train_member(X_train, y[dev_train], args.epochs) for _ in range(args.members)]
            probs.append(np.mean([m.predict(X_val, verbose=0) for m in members], axis=0))
            print('  %-18s seed %d done (%d features, %d member(s))'
                  % (spec, seed, X.shape[1], args.members), flush=True)
        results[spec] = evaluate_arm(y[dev_val], probs, any_prepared.classes, args.threshold)
        if getattr(args, 'sweep', False):
            sweeps[spec] = _average_sweeps([threshold_sweep(y[dev_val], p) for p in probs])

    print()
    print(format_report(results, any_prepared.classes))

    for spec, rows in sweeps.items():
        print('\nTHRESHOLD SWEEP - %s (dev split, mean of %d seed(s), %d member(s) each)'
              % (spec, len(args.seeds), args.members))
        print('%9s %9s %9s %9s %9s %11s %11s' % ('threshold', 'precision', 'recall', 'micro F1',
                                                 'false pos', 'tags / map', 'tags used'))
        for r in rows:
            if round(r['threshold'] * 100) % 2 == 0 or r['threshold'] == args.threshold:
                print('%9.2f %9.3f %9.3f %9.3f %9.0f %11.2f %11.1f%s' % (
                    r['threshold'], r['precision'], r['recall'], r['f1'], r['false_positives'],
                    r['pred_tags_per_map'], r['tags_ever_predicted'],
                    '   <- current' if r['threshold'] == args.threshold else ''))
        print('true tags / map on the dev split: %.2f' % y[dev_val].sum(1).mean())

    if args.sweep_json and sweeps:
        if os.path.dirname(args.sweep_json):
            os.makedirs(os.path.dirname(args.sweep_json), exist_ok=True)
        with open(args.sweep_json, 'w', encoding='utf-8') as f:
            json.dump({'n_dev_val': int(len(dev_val)), 'seeds': args.seeds, 'members': args.members,
                       'true_tags_per_map': float(y[dev_val].sum(1).mean()), 'sweeps': sweeps}, f, indent=2)
        print('wrote %s' % args.sweep_json)
    return 0


def _average_sweeps(runs):
    """Mean of each numeric field across seeds, threshold by threshold."""
    return [{k: float(np.mean([run[i][k] for run in runs])) if k != 'threshold' else runs[0][i][k]
             for k in runs[0][i]} for i in range(len(runs[0]))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    add_arguments(ap)
    return run(ap.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
