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

    rebuild()
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
    train_and_evaluate_ensemble(num_models=args.models)
    return 0


def cmd_export_onnx(args):
    """Convert the trained .keras models to the .onnx files the WPF app loads."""
    from export_to_onnx import convert_models

    if not os.path.exists('ensemble_model_1.keras'):
        print("No trained ensemble found. Run train-ensemble first.")
        return 1

    convert_models()

    # convert_models() skips missing models with a printed warning rather than
    # failing, so verify the outputs actually exist before calling this a
    # success - a green exit code on a partial export would ship a broken app.
    missing = [i for i in range(1, 6) if not os.path.exists(f'ensemble_model_{i}.onnx')]
    if missing:
        print(f"Export incomplete - missing ONNX for model(s): {missing}")
        return 1
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
    p.set_defaults(func=cmd_train_ensemble)

    p = sub.add_parser('export-onnx', help='Convert trained .keras models to .onnx for the app')
    p.set_defaults(func=cmd_export_onnx)

    p = sub.add_parser('predict', help='Predict tags for a single .osu file')
    p.add_argument('--map', required=True, help='Path to a .osu file')
    p.add_argument('--threshold', type=float, default=0.27, help='Confidence cutoff (default: 0.27)')
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser('evaluate', help='Run predictions across a folder of maps')
    p.add_argument('--songs', default='songs', help='Folder of .osu files (default: songs)')
    p.add_argument('--max-maps', type=int, default=10)
    p.add_argument('--threshold', type=float, default=0.27)
    p.set_defaults(func=cmd_evaluate)

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
