#!/usr/bin/env python3
"""
Non-interactive entry point for the osu! beatmap classifier.

main.py is the human interface: it prints a menu and waits for input(). That
makes it impossible to run unattended - in a container with no terminal, the
first prompt raises EOFError and the run dies.

This file exposes the same operations as named subcommands with flags, so every
answer main.py would have asked for is supplied up front:

    python cli.py train-ensemble --models 5
    python cli.py evaluate --max-maps 50 --threshold 0.27

It deliberately contains NO logic of its own - every subcommand is a thin call
into the same functions main.py uses. If behaviour differs between the two,
that is a bug here, not a feature.

Exit codes: 0 on success, non-zero on failure. Automation reads the exit code,
not the printed output, so anything that "failed but printed a message and
returned normally" is invisible to CI. Hence the explicit returns below.
"""
import argparse
import os
import sys

# TensorFlow prints several lines of INFO noise on import. Harmless
# interactively, but it buries the real output in CI logs. Must be set BEFORE
# tensorflow is imported anywhere, which is why it sits at module top.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


# --- Subcommand implementations -------------------------------------------
#
# Every heavy import (tensorflow, the model classes) happens INSIDE the
# function that needs it, not at the top of the file. Importing tensorflow
# takes several seconds, and `cli.py --help` should not pay for it.

def cmd_build_dataset(args):
    """Scrape map metadata + .osu files from the APIs and write ml_dataset.json."""
    from dataset_builder import build_full_dataset, save_dataset

    print(f"Building dataset: {args.max_maps} maps starting at offset {args.offset}")
    data = build_full_dataset(max_maps=args.max_maps, offset=args.offset)

    # build_full_dataset returns the dataset rather than saving it, and returns
    # an empty list on auth failure. Saving is the caller's job; not doing it
    # is what silently discarded an hour of scraping in main.py.
    if not save_dataset(data, args.output):
        print("Dataset build produced no maps. Check ECHO_API_TOKEN and the osu! API credentials.")
        return 1
    return 0


def cmd_rebuild(args):
    """Re-parse the already-downloaded .osu files in downloads/ into a dataset."""
    from rebuild_from_downloaded import rebuild

    # rebuild() used to return None whatever happened - missing token, empty
    # downloads/, nothing matched - and this returned 0 regardless. A scripted
    # `cli.py rebuild && cli.py train-ensemble` would then train on whatever
    # stale dataset was lying around, which is exactly the silent failure the
    # exit-code contract exists to prevent.
    if not rebuild():
        print("Rebuild did not produce a dataset.")
        return 1
    return 0


def cmd_train(args):
    """Train the single model and save beatmap_classifier.pkl."""
    from neural_model import ImprovedBeatmapClassifier

    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}. Run build-dataset or rebuild first.")
        return 1

    ImprovedBeatmapClassifier().train(dataset_filename=args.dataset)
    return 0


def cmd_train_ensemble(args):
    """Train the N-model ensemble that the shipped ONNX files come from."""
    from ensemble_evaluator import train_and_evaluate_ensemble

    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}. Run build-dataset or rebuild first.")
        return 1

    print(f"Training {args.models} models in sequence. This is the slow one.")

    # --dataset used to be validated here and then ignored by the trainer, which
    # hard-coded ml_dataset.json. Pass every flag through explicitly.
    result = train_and_evaluate_ensemble(
        num_models=args.models,
        dataset=args.dataset,
        epochs=args.epochs,
        train_seed=args.train_seed,
        out_dir=args.out_dir)

    if result is None:
        print("Training failed - no models were produced.")
        return 1

    print(f"\nArtifacts written to: {result['out_dir']}")
    return 0


def cmd_export_onnx(args):
    """Convert the trained .keras models to the .onnx files the WPF app loads."""
    from export_to_onnx import convert_models
    from extract_config import extract_config

    model_dir = args.model_dir
    out_dir = args.out_dir or model_dir

    if not os.path.exists(os.path.join(model_dir, 'ensemble_model_1.keras')):
        print(f"No trained ensemble found in {model_dir}. Run train-ensemble first.")
        return 1

    convert_models(model_dir=model_dir, out_dir=out_dir)

    # convert_models() skips missing models with a printed warning rather than
    # failing, so verify the outputs actually exist before calling this a
    # success - a green exit code on a partial export would ship a broken app.
    missing = [i for i in range(1, 6)
               if not os.path.exists(os.path.join(out_dir, f'ensemble_model_{i}.onnx'))]
    if missing:
        print(f"Export incomplete - missing ONNX for model(s): {missing}")
        return 1

    # The app needs six files, not five. model_config.json carries the scaler
    # constants and tag list, and it has to come from the SAME artifacts as the
    # .onnx files - a config from a different run standardises the features with
    # the wrong numbers and corrupts every prediction without erroring. It used
    # to be a separate script you had to remember to run; now it cannot be
    # forgotten, and a missing one fails the export.
    config_path = extract_config(model_dir=model_dir,
                                 out_path=os.path.join(out_dir, 'model_config.json'))
    if config_path is None or not os.path.exists(config_path):
        print("Export incomplete - model_config.json was not written. "
              "The app cannot load a model without it.")
        return 1

    print(f"\nExported 5 ONNX models + model_config.json to {os.path.abspath(out_dir)}")
    return 0


def cmd_predict(args):
    """Predict tags for one .osu file and print them."""
    from neural_model import ImprovedBeatmapClassifier

    if not os.path.exists(args.map):
        print(f"Map not found: {args.map}")
        return 1

    classifier = ImprovedBeatmapClassifier()

    if os.path.exists('ensemble_model_1.keras'):
        from ensemble_evaluator import load_ensemble_assets, predict_with_ensemble
        assets = load_ensemble_assets()
        tags = predict_with_ensemble(args.map, args.threshold, assets, classifier)
    else:
        tags = classifier.predict_tags(args.map, threshold=args.threshold)

    print(f"\nPredicted tags: {tags}")
    return 0


def cmd_evaluate(args):
    """Run predictions over a folder of maps - the closest thing to a test."""
    from neural_model import ImprovedBeatmapClassifier

    # --holdout is the measured path: score a model on the fixed evaluation
    # split and record the numbers. Without it this command behaves exactly as
    # it always has, printing predicted tags for a folder of maps, because that
    # qualitative spot-check is still the quickest way to see if a model has
    # gone obviously wrong.
    if args.holdout:
        return _evaluate_holdout(args)

    if not os.path.isdir(args.songs):
        print(f"Songs folder not found: {args.songs}")
        return 1

    if os.path.exists('ensemble_model_1.keras'):
        from ensemble_evaluator import test_multiple_maps_with_ensemble
        test_multiple_maps_with_ensemble(max_maps=args.max_maps, threshold=args.threshold)
    else:
        ImprovedBeatmapClassifier().test_multiple_maps(
            songs_folder=args.songs, threshold=args.threshold, max_maps=args.max_maps)
    return 0


def _evaluate_holdout(args):
    """Score a model directory on the fixed split and log the run to MLflow."""
    import split as split_mod
    from metrics_report import format_summary
    from scoring import count_models, score_on_holdout

    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}.")
        return 1
    if count_models(args.model_dir) == 0:
        print(f"No ensemble_model_*.keras found in: {args.model_dir}")
        return 1

    prepared = split_mod.prepare_dataset(args.dataset)
    sp = split_mod.fixed_split(prepared)
    split_mod.write_split_manifest(prepared, sp)

    try:
        summary, per_tag, _probs = score_on_holdout(
            args.model_dir, prepared, sp, threshold=args.threshold)
    except (ValueError, FileNotFoundError) as e:
        print(f"Evaluation failed: {e}")
        return 1

    print(f"\nHoldout: {summary['n_samples']} maps, split hash {sp.split_hash[:16]}")
    print(format_summary(summary))
    print("\nTop 10 tags by F1:")
    print(per_tag.head(10).to_string(index=False))

    if args.no_log:
        return 0

    from tracking import log_evaluation

    run_id = log_evaluation(
        summary, per_tag,
        params={
            'split_seed': sp.seed,
            'test_size': sp.test_size,
            'threshold': args.threshold,
            'n_features': int(prepared.X.shape[1]),
            'n_labels': len(prepared.classes),
            'dataset': args.dataset,
            'dataset_sha256': prepared.dataset_sha,
            'split_hash': sp.split_hash,
            'model_dir': args.model_dir,
            'num_models': summary['num_models'],
        },
        extra_artifacts=['split_manifest.json',
                         os.path.join(args.model_dir, 'model_config.json')],
        run_name=args.run_name or f'evaluate-{os.path.basename(os.path.abspath(args.model_dir))}')
    print(f"\nLogged MLflow run: {run_id}")
    return 0


def cmd_promote(args):
    """Compare a candidate against the champion and promote only if it holds up."""
    import shutil

    import promote as gate
    import registry
    import split as split_mod
    from metrics_report import format_summary
    from scoring import count_models, score_on_holdout

    if count_models(args.candidate) == 0:
        print(f"No candidate models found in: {args.candidate}")
        return 1
    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}.")
        return 1

    tolerance = args.tolerance if args.tolerance is not None else gate.DEFAULT_TOLERANCE
    if tolerance is None:
        print("No --tolerance given and no calibrated default is set. "
              "Run the seed calibration first; the gate will not guess.")
        return 1

    prepared = split_mod.prepare_dataset(args.dataset)
    sp = split_mod.fixed_split(prepared)

    print(f"Scoring candidate on the fixed split ({len(sp.test_idx)} maps, "
          f"hash {sp.split_hash[:16]})...")
    try:
        cand_summary, cand_per_tag, _ = score_on_holdout(
            args.candidate, prepared, sp, threshold=args.threshold)
    except (ValueError, FileNotFoundError) as e:
        print(f"Could not score candidate: {e}")
        return 1
    print(f"Candidate: {format_summary(cand_summary)}")

    # Re-score the champion on TODAY's split rather than trusting its stored
    # metric. If the dataset moved, the stored number describes a different test
    # set and comparing against it would be meaningless.
    champion = registry.get_champion()
    champion_f1 = None
    champion_dir = None
    if champion is not None:
        version, run_id, stored_f1 = champion
        print(f"Champion: registered version {version} (run {run_id[:8]}), "
              f"stored macro_f1={stored_f1}")
        try:
            champion_dir = registry.download_champion_ensemble()
            champ_summary, _, _ = score_on_holdout(
                champion_dir, prepared, sp, threshold=args.threshold)
            champion_f1 = champ_summary['macro_f1']
            print(f"Champion re-scored on this split: {format_summary(champ_summary)}")
        except Exception as e:                      # noqa: BLE001
            print(f"Could not re-score the champion: {e}")
            registry.cleanup(champion_dir)
            return 1
    else:
        print("No champion registered yet.")

    best_ever = registry.get_best_ever()
    verdict = gate.decide(
        candidate_f1=cand_summary['macro_f1'],
        champion_f1=champion_f1,
        best_ever_f1=best_ever,
        tolerance=tolerance)

    print(f"\n{verdict}")
    registry.cleanup(champion_dir)

    if not verdict.promote:
        # Non-zero exit is the entire point: this is what stops the Prefect flow
        # before export-onnx and what fails a CI pipeline.
        return 2

    run_id, version = registry.register_ensemble(
        args.candidate, cand_summary, cand_per_tag,
        params={
            'split_seed': sp.seed,
            'threshold': args.threshold,
            'dataset_sha256': prepared.dataset_sha,
            'split_hash': sp.split_hash,
            'tolerance': tolerance,
        },
        tags={'promoted_from': args.candidate, 'gate_reason': verdict.reason},
        run_name=args.run_name or f'promote-{os.path.basename(os.path.abspath(args.candidate))}')

    if version is None:
        print("Registered the run but could not resolve a model version.")
        return 1

    registry.set_champion(version)
    print(f"Registered version {version} and moved the 'champion' alias to it "
          f"(run {run_id}).")

    # Copy artifacts to --root-dir. Defaults to the repo root, which is where
    # export-onnx and the app workflow look; verification runs point it at a
    # temp directory so a demo never replaces the shipped models.
    root = args.root_dir
    if root and os.path.abspath(root) != os.path.abspath(args.candidate):
        os.makedirs(root, exist_ok=True)
        copied = []
        for name in sorted(os.listdir(args.candidate)):
            if name.endswith('.keras') or name.endswith('.pkl'):
                shutil.copy2(os.path.join(args.candidate, name), os.path.join(root, name))
                copied.append(name)
        print(f"Copied {len(copied)} artifact(s) into {os.path.abspath(root)}")
    return 0


def cmd_drift(args):
    """Compare a folder of new maps against the training feature distribution."""
    import split as split_mod
    from drift import build_report, features_from_folder

    if not os.path.isdir(args.maps):
        print(f"Maps folder not found: {args.maps}")
        return 1
    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}.")
        return 1

    try:
        current_X, names, skipped = features_from_folder(args.maps, max_maps=args.max_maps)
    except (ValueError, NotADirectoryError) as e:
        print(f"Could not extract features: {e}")
        return 1

    print(f"Extracted features from {len(names)} map(s) in {args.maps}"
          + (f", skipped {len(skipped)}" if skipped else ""))
    for filename, why in skipped[:5]:
        print(f"  skipped {filename}: {why}")

    prepared = split_mod.prepare_dataset(args.dataset)
    summary, path = build_report(prepared.X, current_X, out_path=args.out)

    if not summary.get('drift_summary_parsed'):
        print("Report written, but the drift counts could not be read out of "
              "Evidently's result - reporting them as unknown rather than "
              "guessing a number.")
    else:
        share = summary['drift_share']
        print(f"\nDrifted columns: {summary['drifted_columns']} of "
              f"{summary['n_features']}  (share {share:.3f}, per-column "
              f"threshold from Evidently's default preset)")
        if summary.get('top_drifted'):
            print("Most drifted features:")
            for name, score in summary['top_drifted']:
                print(f"  {name:<34} {score:.3f}")
    print(f"Report: {os.path.abspath(path)}")

    if not args.no_log:
        from tracking import log_evaluation
        metrics = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
        run_id = log_evaluation(
            metrics, None,
            params={'maps_folder': args.maps, 'n_current': summary['n_current'],
                    'dataset_sha256': prepared.dataset_sha},
            extra_artifacts=[path],
            run_name=args.run_name or f'drift-{os.path.basename(os.path.abspath(args.maps))}')
        print(f"Logged MLflow run: {run_id}")
    return 0


def cmd_pipeline(args):
    """Run the whole train -> evaluate -> gate -> export flow with Prefect."""
    from flows.pipeline import training_pipeline

    state = training_pipeline(
        dataset=args.dataset,
        models=args.models,
        epochs=args.epochs,
        train_seed=args.train_seed,
        tolerance=args.tolerance,
        threshold=args.threshold,
        candidate_dir=args.candidate_dir,
        root_dir=args.root_dir,
        build_dataset=args.build_dataset,
        max_maps=args.max_maps,
        return_state=True)

    # Prefect swallows task exceptions into run state; translate that back into
    # the exit-code contract the rest of this CLI keeps.
    if state.is_failed() or state.is_crashed():
        print(f"\nPipeline did not complete: {state.message}")
        return 1
    return 0


# --- Argument wiring -------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog='cli.py',
        description='Non-interactive interface to the osu! beatmap classifier.')

    # Subparsers turn the first positional word into a command, git-style:
    # `cli.py train` vs `cli.py predict`. required=True means running with no
    # arguments prints help and exits non-zero instead of doing something.
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('build-dataset', help='Scrape maps + tags into ml_dataset.json (slow, needs .env)')
    p.add_argument('--max-maps', type=int, default=5000, help='How many maps to process (default: 5000)')
    p.add_argument('--offset', type=int, default=0, help='Skip this many maps in the list, to resume a run')
    p.add_argument('--output', default='ml_dataset.json', help='Where to write the dataset')
    p.set_defaults(func=cmd_build_dataset)

    p = sub.add_parser('rebuild', help='Rebuild ml_dataset.json from the local downloads/ folder (no network)')
    p.set_defaults(func=cmd_rebuild)

    p = sub.add_parser('train', help='Train the single model')
    p.add_argument('--dataset', default='ml_dataset.json')
    p.set_defaults(func=cmd_train)

    p = sub.add_parser('train-ensemble', help='Train the 5-model ensemble (slow)')
    p.add_argument('--models', type=int, default=5, help='Number of models in the ensemble (default: 5)')
    p.add_argument('--dataset', default='ml_dataset.json')
    p.add_argument('--epochs', type=int, default=100,
                   help='Max epochs per model, early stopping still applies (default: 100)')
    p.add_argument('--train-seed', type=int, default=None,
                   help='Seed for weight init and shuffling. Does NOT affect the '
                        'evaluation split, which is frozen at seed 42.')
    p.add_argument('--out-dir', default='.',
                   help='Where to write models, scaler and binarizer. Use '
                        'candidates/<name> to train without touching the models '
                        'currently in the repo root (default: .)')
    p.set_defaults(func=cmd_train_ensemble)

    p = sub.add_parser('export-onnx',
                       help='Convert trained .keras models to the 5 .onnx files + '
                            'model_config.json the app loads')
    p.add_argument('--model-dir', default='.', help='Where the .keras models live (default: .)')
    p.add_argument('--out-dir', default=None, help='Where to write outputs (default: --model-dir)')
    p.set_defaults(func=cmd_export_onnx)

    p = sub.add_parser('predict', help='Predict tags for a single .osu file')
    p.add_argument('--map', required=True, help='Path to a .osu file')
    p.add_argument('--threshold', type=float, default=0.27, help='Confidence cutoff (default: 0.27)')
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser('evaluate', help='Run predictions across a folder of maps, '
                                        'or score a model on the fixed split with --holdout')
    p.add_argument('--songs', default='songs', help='Folder of .osu files (default: songs)')
    p.add_argument('--max-maps', type=int, default=10)
    p.add_argument('--threshold', type=float, default=0.27)
    p.add_argument('--holdout', action='store_true',
                   help='Score a model on the fixed evaluation split and report '
                        'micro/macro F1 instead of printing per-map tags')
    p.add_argument('--model-dir', default='.', help='Model directory to score (default: .)')
    p.add_argument('--dataset', default='ml_dataset.json')
    p.add_argument('--run-name', default=None, help='Name for the MLflow run')
    p.add_argument('--no-log', action='store_true', help='Skip MLflow logging')
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser('promote', help='Gate a candidate against the champion and '
                                       'promote it only if quality holds up')
    p.add_argument('--candidate', required=True, help='Directory holding the candidate ensemble')
    p.add_argument('--dataset', default='ml_dataset.json')
    p.add_argument('--threshold', type=float, default=0.27)
    p.add_argument('--tolerance', type=float, default=None,
                   help='How much lower than the baseline macro F1 is acceptable. '
                        'Defaults to the calibrated value in promote.py.')
    p.add_argument('--root-dir', default='.',
                   help='Where a promoted model is copied (default: . - the repo '
                        'root the app workflow reads). Point at a temp dir to '
                        'exercise promotion without replacing the live models.')
    p.add_argument('--run-name', default=None)
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser('drift', help='Compare a folder of .osu files against the '
                                     'training feature distribution')
    p.add_argument('--maps', required=True, help='Folder of .osu files to check')
    p.add_argument('--dataset', default='ml_dataset.json')
    p.add_argument('--out', default='drift_report.html')
    p.add_argument('--max-maps', type=int, default=None)
    p.add_argument('--run-name', default=None)
    p.add_argument('--no-log', action='store_true', help='Skip MLflow logging')
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser('pipeline', help='Run train -> evaluate -> promote -> export '
                                        'as one Prefect flow')
    p.add_argument('--dataset', default='ml_dataset.json')
    p.add_argument('--models', type=int, default=5)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--train-seed', type=int, default=None)
    p.add_argument('--tolerance', type=float, default=None)
    p.add_argument('--threshold', type=float, default=0.27)
    p.add_argument('--candidate-dir', default=None,
                   help='Where to train the candidate (default: candidates/<timestamp>)')
    p.add_argument('--root-dir', default='.',
                   help='Where a promoted model lands; use a temp dir to keep the '
                        'shipped models untouched')
    p.add_argument('--build-dataset', action='store_true',
                   help='Scrape a fresh dataset first (needs network and .env)')
    p.add_argument('--max-maps', type=int, default=5000,
                   help='Only used with --build-dataset')
    p.set_defaults(func=cmd_pipeline)

    return parser


def main():
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == '__main__':
    # sys.exit() with the returned code is what makes this usable from a script:
    # `cli.py train-ensemble && cli.py export-onnx` only exports if training
    # actually succeeded. Note there is deliberately no try/except here - an
    # unhandled exception should crash loudly with a traceback and a non-zero
    # code, not be swallowed into a tidy error message that hides the cause.
    sys.exit(main())
