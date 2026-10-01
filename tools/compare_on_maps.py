"""
Two models' tags, side by side, on maps you know.

Numbers say whether tags land on the right maps on average; this shows it map
by map, which is how a misplaced tag is actually noticed. For each .osu file it
prints the tags both models give, the tags only one of them gives, and - when
the map is in the dataset - the community's tags after the label policy, so a
disagreement can be read against what players actually tagged.

    python -m tools.compare_on_maps --model-dir . --model-dir candidates/v2-seed-1

Each model is fed the feature version it was trained on (feature_meta.json).
Tags are what a player would see: the shared threshold rule and the redundancy
suppression (labels.py) are applied. The 'streams' override is not.
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from mlops.labels import THRESHOLD, predicted, suppress_redundant  # noqa: E402


def _features(path, version, classifier):
    from osu_parser import OsuFileParser

    parser = OsuFileParser(path)
    parser.read_file()
    hit_objects = parser.extract_raw_hit_objects()
    sections = classifier.split_beatmap_into_sections(hit_objects)
    if not sections:
        return None, parser
    if version == 2:
        from features_v2 import extract_features_v2
        return extract_features_v2(hit_objects, parser.get_difficulty(),
                                   parser.get_timing_points()), parser
    return classifier._aggregate_features_for_map(sections), parser


def _community_tags(dataset_path):
    """beatmap_id -> labels after the policy. Loads the whole dataset once (~30 s)."""
    from mlops.labels import apply_label_policy

    if not os.path.exists(dataset_path):
        return {}
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {str(s.get('beatmap_id')): apply_label_policy(s.get('tags') or []) for s in data}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--model-dir', action='append', required=True,
                    help='Give exactly two: the baseline first, then the one to compare')
    ap.add_argument('--maps', default='songs')
    ap.add_argument('--dataset', default='ml_dataset.json',
                    help="Looked up for each map's community tags (skipped if absent)")
    ap.add_argument('--threshold', type=float, default=THRESHOLD)
    args = ap.parse_args()
    if len(args.model_dir) != 2:
        ap.error('give --model-dir exactly twice')

    from mlops.labels import apply_label_policy
    from mlops.scoring import ensemble_probabilities, load_ensemble
    from mlops.split import model_feature_version
    from neural_model import ImprovedBeatmapClassifier

    classifier = ImprovedBeatmapClassifier()
    models = []
    for d in args.model_dir:
        ens, scaler, binarizer = load_ensemble(d)
        models.append((d, ens, scaler, list(binarizer.classes_), model_feature_version(d)))
    truth = _community_tags(args.dataset)

    a_name, b_name = [os.path.basename(os.path.abspath(d)) or d for d in args.model_dir]
    files = sorted(f for f in os.listdir(args.maps) if f.endswith('.osu'))
    for filename in files:
        path = os.path.join(args.maps, filename)
        tag_sets, parser = [], None
        for d, ens, scaler, classes, version in models:
            vec, parser = _features(path, version, classifier)
            if vec is None:
                tag_sets.append(None)
                continue
            probs = ensemble_probabilities(ens, scaler, np.asarray(vec).reshape(1, -1))[0]
            # Compare on the current label space, whatever policy each model was trained under.
            tag_sets.append(set(suppress_redundant(apply_label_policy(
                [c for c, p in zip(classes, probs) if predicted(p, args.threshold)]))))

        print('\n' + filename[:-4])
        if tag_sets[0] is None:
            print('  too short to measure')
            continue
        both = sorted(tag_sets[0] & tag_sets[1])
        print('  both          : %s' % ', '.join(both))
        print('  only %-9s: %s' % (a_name[:9], ', '.join(sorted(tag_sets[0] - tag_sets[1])) or '-'))
        print('  only %-9s: %s' % (b_name[:9], ', '.join(sorted(tag_sets[1] - tag_sets[0])) or '-'))
        bid = parser.get_beatmap_id() if parser else None
        if bid and bid in truth:
            print('  community     : %s' % ', '.join(truth[bid]))


if __name__ == '__main__':
    main()
