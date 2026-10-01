"""
Every tag's probability, for every map in a folder, from one or more models.

Choosing a threshold means seeing what sits just below it and just above it.
The predict path prints only what cleared the threshold, which hides exactly
that. This prints and saves the lot:

  terminal   per map and model, every tag at or above --show, highest first,
             '*' marking the ones the threshold would predict and '~' the
             ones predicted but hidden as redundant (labels.SUPPRESSED_BY);
             plus the community's tags when the map is in the dataset
  CSV        one row per (map, model), one column per tag - opens in a spreadsheet
  JSON       the same data, for plotting or an interactive threshold explorer

    python cli.py tag-probabilities --model-dir shipped=. --model-dir v2=candidates/v2-seed-2
    python -m tools.map_probabilities ...      (the same command, outside the CLI)

Each model is fed the feature version it was trained on, and its outputs are
put on the current label space (labels.py), so an old and a new model line up
column for column. A map's 'split' says whether the models trained on it: its
probabilities are only an honest guess when it is 'holdout' or 'unseen'.
"""
import argparse
import csv
import json
import os

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from mlops.labels import THRESHOLD, predicted, suppress_redundant  # noqa: E402


def add_arguments(ap):
    """Shared by `python -m tools.map_probabilities` and `cli.py tag-probabilities`."""
    ap.add_argument('--model-dir', action='append', required=True,
                    help='Repeatable. Optionally NAME=DIR to label it')
    ap.add_argument('--maps', default='songs')
    ap.add_argument('--dataset', default='ml_dataset.json',
                    help="For each map's community tags (skipped if absent)")
    ap.add_argument('--threshold', type=float, default=THRESHOLD)
    ap.add_argument('--show', type=float, default=0.10,
                    help='Terminal: list tags at or above this probability (default 0.10)')
    ap.add_argument('--csv', default=os.path.join('reports', 'map_probabilities.csv'))
    ap.add_argument('--json', default=os.path.join('reports', 'map_probabilities.json'))


def run(args):
    """Returns an exit code, like every cli.py subcommand: 0 on success."""
    from mlops.labels import canonical, project_probabilities
    from mlops.scoring import ensemble_probabilities, load_ensemble
    from mlops.split import MANIFEST_PATH, model_feature_version
    from neural_model import ImprovedBeatmapClassifier
    from mlops.scoring import count_models
    from tools.compare_on_maps import _community_tags, _features

    if not os.path.isdir(args.maps):
        print('Maps folder not found: %s' % args.maps)
        return 1
    for spec in args.model_dir:
        d = spec.rpartition('=')[2]
        if count_models(d) == 0:
            print('No ensemble_model_*.keras found in: %s' % d)
            return 1

    classifier = ImprovedBeatmapClassifier()
    models = []
    for spec in args.model_dir:
        name, _, d = spec.rpartition('=')
        name = name or os.path.basename(os.path.abspath(d)) or d
        ens, scaler, binarizer = load_ensemble(d)
        classes = list(binarizer.classes_)
        labels = sorted({canonical(c) for c in classes} - {None})
        models.append({'name': name, 'dir': d, 'ens': ens, 'scaler': scaler, 'labels': labels,
                       'projection': project_probabilities(classes, labels),
                       'version': model_feature_version(d)})
    tags = models[0]['labels']
    for m in models[1:]:
        if m['labels'] != tags:
            print('%s and %s have different label spaces after the policy'
                  % (models[0]['name'], m['name']))
            return 1

    community = _community_tags(args.dataset)
    holdout = set()
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding='utf-8') as f:
            holdout = set(json.load(f)['holdout_beatmap_ids'])

    maps = []
    for filename in sorted(f for f in os.listdir(args.maps) if f.endswith('.osu')):
        path = os.path.join(args.maps, filename)
        entry = {'map': filename[:-4], 'beatmap_id': None, 'split': 'unseen',
                 'community': None, 'probs': {}}
        for m in models:
            vec, parser = _features(path, m['version'], classifier)
            if vec is None:
                break
            raw = ensemble_probabilities(m['ens'], m['scaler'], np.asarray(vec).reshape(1, -1))
            entry['probs'][m['name']] = [round(float(p), 4) for p in m['projection'].apply(raw)[0]]
            bid = parser.get_beatmap_id()
            entry['beatmap_id'] = bid
        if not entry['probs']:
            continue
        bid = entry['beatmap_id']
        if bid in community:
            entry['community'] = community[bid]
            entry['split'] = 'holdout' if bid in holdout else 'train'
        maps.append(entry)

    # --- terminal ---
    for entry in maps:
        print('\n%s   [%s]' % (entry['map'], entry['split']))
        for m in models:
            probs = entry['probs'][m['name']]
            shown = sorted(((p, t) for t, p in zip(tags, probs) if p >= args.show), reverse=True)
            kept = set(suppress_redundant([t for t, p in zip(tags, probs) if predicted(p, args.threshold)]))

            def mark(tag, p):
                if not predicted(p, args.threshold):
                    return ''
                return '*' if tag in kept else '~'

            print('  %-14s %s' % (m['name'][:14], '  '.join(
                '%s %.2f%s' % (t, p, mark(t, p)) for p, t in shown) or '-'))
        if entry['community'] is not None:
            print('  %-14s %s' % ('community', ', '.join(entry['community'])))
    print('\n* = predicted at the %.2f threshold; ~ = predicted but hidden next to a more specific tag. '
          'Tags under %.2f not shown; the CSV has all %d.' % (args.threshold, args.show, len(tags)))

    # --- files ---
    for path in (args.csv, args.json):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(args.csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['map', 'beatmap_id', 'split', 'model', 'community_tags'] + tags)
        for entry in maps:
            for m in models:
                w.writerow([entry['map'], entry['beatmap_id'], entry['split'], m['name'],
                            ';'.join(entry['community'] or [])] + entry['probs'][m['name']])
    with open(args.json, 'w', encoding='utf-8') as f:
        json.dump({'threshold': args.threshold, 'tags': tags,
                   'models': [{'name': m['name'], 'dir': m['dir'], 'feature_version': m['version']}
                              for m in models],
                   'maps': maps}, f, indent=1)
    print('wrote %s and %s' % (args.csv, args.json))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    add_arguments(ap)
    return run(ap.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
