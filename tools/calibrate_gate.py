"""
Recalibrate the promotion gate: measure training noise, derive the tolerance.

The constants in mlops/promote.py (MICRO_F1_SIGMA, TOLERANCE_K,
NEVER_PREDICTED_ALLOWANCE) are measurements, not choices. This script is how
they were measured and the only supported way to change them - editing them by
hand to let a particular candidate through defeats the point of having a gate.

    python -m tools.calibrate_gate --seeds 1 2 3 4 5 6 7 8 9 10

It trains the identical configuration once per seed, varying ONLY --train-seed
(the evaluation split stays frozen at seed 42), scores every run through the
same code path on the same holdout, and reports the spread of each candidate
gate metric.

Reads the tolerance off standard deviation, not the max pairwise gap. The gap is
an order statistic: its expectation grows roughly as sigma*sqrt(2 ln n), so it
widens every time a seed is added and never settles. The first calibration of
this gate used the gap over 4 runs, and a later run promptly fell outside it.
Both numbers are printed so that mistake stays visible.

Existing candidate directories are reused rather than retrained, so adding seeds
to an earlier calibration is cheap. Nothing here writes to the repo root or the
real registry.
"""
import argparse
import json
import os
import statistics
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')


def _metrics_for(model_dir, prepared, split, support, floor, threshold):
    """Every candidate gate metric for one trained ensemble."""
    import numpy as np
    from sklearn.metrics import f1_score

    from mlops.scoring import ensemble_probabilities, load_ensemble

    models, scaler, _ = load_ensemble(model_dir)
    y_true = prepared.y[split.test_idx]
    probs = ensemble_probabilities(models, scaler, prepared.X[split.test_idx])
    y_pred = (probs >= threshold).astype(int)

    per_tag = f1_score(y_true, y_pred, average=None, zero_division=0)
    never = y_pred.sum(axis=0) == 0
    included = support >= floor

    return {
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'macro_f1_supported': float(per_tag[included].mean()),
        'never_predicted': int(never.sum()),
        'never_predicted_supported': int((never & included).sum()),
        'never_predicted_rare': int(np.count_nonzero(never & ~included)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    ap.add_argument('--dataset', default='ml_dataset.json')
    ap.add_argument('--models', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--threshold', type=float, default=0.27)
    ap.add_argument('--k', type=float, default=4.0,
                    help='Tolerance multiple of sigma (default: 4, see promote.py)')
    ap.add_argument('--champion-dir', default='.',
                    help='Scored as an extra same-config sample and as the baseline')
    ap.add_argument('--out', default='gate_calibration.json')
    ap.add_argument('--retrain', action='store_true',
                    help='Retrain seeds even if candidates/seed-N already exists')
    args = ap.parse_args()

    from ensemble_evaluator import train_and_evaluate_ensemble
    from mlops import split as split_mod
    from mlops.metrics_report import SUPPORT_FLOOR

    prepared = split_mod.prepare_dataset(args.dataset)
    sp = split_mod.fixed_split(prepared)
    support = prepared.y[sp.test_idx].sum(axis=0)
    print('Holdout %d maps, %d labels, split %s'
          % (len(sp.test_idx), len(prepared.classes), sp.split_hash[:16]))
    print('support floor %d -> %d of %d tags included'
          % (SUPPORT_FLOOR, int((support >= SUPPORT_FLOOR).sum()), len(support)))

    runs = {}
    for seed in args.seeds:
        out_dir = os.path.join('candidates', 'seed-%d' % seed)
        if args.retrain or not os.path.isdir(out_dir):
            print('\n=== training seed %d -> %s ===' % (seed, out_dir), flush=True)
            t0 = time.time()
            if train_and_evaluate_ensemble(
                    num_models=args.models, dataset=args.dataset, epochs=args.epochs,
                    train_seed=seed, out_dir=out_dir) is None:
                raise SystemExit('seed %d failed to train' % seed)
            print('  trained in %.0fs' % (time.time() - t0), flush=True)
        runs[seed] = _metrics_for(out_dir, prepared, sp, support,
                                  SUPPORT_FLOOR, args.threshold)
        print('seed %-3d micro_f1=%.6f macro_f1=%.6f never(supported)=%d'
              % (seed, runs[seed]['micro_f1'], runs[seed]['macro_f1'],
                 runs[seed]['never_predicted_supported']), flush=True)

    champion = _metrics_for(args.champion_dir, prepared, sp, support,
                            SUPPORT_FLOOR, args.threshold)
    print('champion micro_f1=%.6f macro_f1=%.6f never(supported)=%d'
          % (champion['micro_f1'], champion['macro_f1'],
             champion['never_predicted_supported']))

    metrics = ['micro_f1', 'macro_f1', 'macro_f1_supported', 'weighted_f1']
    print('\n%-22s %9s %9s %9s %9s %9s %8s' % (
        'metric', 'mean', 'sigma', 'min', 'max', 'gap', 'gap/sig'))
    print('-' * 80)
    spreads = {}
    for m in metrics:
        vals = [runs[s][m] for s in sorted(runs)]
        sigma = statistics.stdev(vals)
        gap = max(vals) - min(vals)
        spreads[m] = {'mean': statistics.fmean(vals), 'stdev': sigma,
                      'min': min(vals), 'max': max(vals), 'max_pairwise_gap': gap,
                      'gap_over_stdev': gap / sigma if sigma else 0.0}
        print('%-22s %9.6f %9.6f %9.6f %9.6f %9.6f %8.2f'
              % (m, spreads[m]['mean'], sigma, min(vals), max(vals), gap,
                 spreads[m]['gap_over_stdev']))

    print('\nFor n=%d the expected range of a normal is ~%.2f sigma; ratios near'
          % (len(runs), 3.08 if len(runs) == 10 else 3.0))
    print('that mean the spread is ordinary noise rather than one odd run.')

    never = [runs[s]['never_predicted_supported'] for s in sorted(runs)]
    allowance = max(0, max(never) - champion['never_predicted_supported'])
    print('\nnever_predicted_supported across seeds: %s (champion %d, sigma %.2f)'
          % (never, champion['never_predicted_supported'],
             statistics.stdev(never) if len(never) > 1 else 0.0))
    print('  smallest allowance admitting every run: +%d' % allowance)

    gate_metric = 'micro_f1'
    sigma = spreads[gate_metric]['stdev']
    tolerance = args.k * sigma
    worst = min(runs[s][gate_metric] for s in runs)
    floor = min(champion[gate_metric], min(runs[s][gate_metric] for s in runs))
    print('\n=== derived gate constants (%s) ===' % gate_metric)
    print('  MICRO_F1_SIGMA           = %.6f' % sigma)
    print('  TOLERANCE_K              = %g' % args.k)
    print('  DEFAULT_TOLERANCE        = %.6f' % tolerance)
    print('  NEVER_PREDICTED_ALLOWANCE= %d' % max(allowance, 1))
    print('  worst run %.6f vs champion floor %.6f -> %s'
          % (worst, champion[gate_metric] - tolerance,
             'accepted' if worst >= champion[gate_metric] - tolerance
             else 'REJECTED (raise k)'))
    print('  (lowest observed score overall: %.6f)' % floor)

    payload = {'runs': runs, 'champion': champion, 'spreads': spreads,
               'k': args.k, 'gate_metric': gate_metric,
               'derived_tolerance': tolerance,
               'never_predicted_allowance': max(allowance, 1),
               'support_floor': SUPPORT_FLOOR, 'split_hash': sp.split_hash,
               'dataset_sha256': prepared.dataset_sha}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print('\nwrote %s' % args.out)


if __name__ == '__main__':
    main()
