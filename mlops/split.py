"""
The single source of truth for turning a dataset into a train/holdout split.

Why this file exists: the evaluation set used to be an implementation detail
buried in ensemble_evaluator.py - a `train_test_split(..., random_state=42)`
whose result nothing recorded. That is reproducible only as long as nobody
notices, because the split is POSITIONAL: it selects rows out of whatever order
the dataset happened to be in. Rebuild ml_dataset.json, or reorder it, and the
"fixed" holdout quietly becomes a different set of maps - so a model scored
after the rebuild is not comparable to one scored before it, and a promotion
gate comparing the two is meaningless.

So the split is still seeded with 42 and still produces exactly the same rows it
always did, but now it is RECORDED BY BEATMAP ID. split_manifest.json holds the
sorted holdout ids and a hash over them. If the dataset changes underneath, the
hash changes and you find out, instead of comparing two models on two different
test sets and trusting the number.

Deliberately preserved from the original pipeline, NOT bugs introduced here:
  * SPLIT_SEED is 42 and test_size is 0.2, matching the shipped models.
  * The scaler is fit on ALL rows before splitting, which leaks the holdout
    feature distribution into standardisation. Fixing that would change
    scaler_mean/scaler_scale, hence model_config.json, hence C# parity - out of
    scope. It applies identically to every model the gate compares, so the
    comparison stays fair even though the absolute numbers are mildly
    optimistic. See VERIFIED.md.
"""
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

# The seed that defines the evaluation set. This is frozen. It is NOT a tuning
# knob, and it is NOT the training seed - see train_seed in ensemble_evaluator,
# which varies weight init while this stays put. Changing this value invalidates
# every metric already in the registry, because they would no longer describe
# the same holdout.
SPLIT_SEED = 42
TEST_SIZE = 0.2

CACHE_DIR = '.cache'
MANIFEST_PATH = 'split_manifest.json'


@dataclass
class Prepared:
    """Everything downstream needs, derived once from a dataset file."""
    X: np.ndarray            # (n, 90) UNSCALED features
    y: np.ndarray            # (n, n_labels) binary
    classes: list            # label names, binarizer order
    ids: list                # beatmap_id per row, same order as X
    tag_lists: list          # raw tags per row, for re-binarising elsewhere
    dataset_sha: str
    source_path: str


@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray
    holdout_ids: list        # sorted
    split_hash: str          # sha256 over the sorted holdout ids
    seed: int = SPLIT_SEED
    test_size: float = TEST_SIZE


def dataset_sha256(path):
    """Content hash of the dataset file, streamed so a 528 MB file is fine."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _extract_from_json(path):
    """
    Re-implements exactly the loop in train_and_evaluate_ensemble: keep samples
    that have both tags and hit_objects, split into sections, aggregate. The
    filter conditions matter - `tags and hit_objects` here vs `tags` alone in
    neural_model.train() is why those two produce different row counts.
    """
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    from neural_model import ImprovedBeatmapClassifier

    classifier = ImprovedBeatmapClassifier()
    with open(path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    X, tag_lists, ids = [], [], []
    for sample in dataset:
        if sample.get('tags') and sample.get('hit_objects'):
            sections = classifier.split_beatmap_into_sections(sample['hit_objects'])
            if not sections:
                continue
            vec = classifier._aggregate_features_for_map(sections)
            if vec is not None:
                X.append(vec)
                tag_lists.append(sample['tags'])
                ids.append(str(sample.get('beatmap_id', 'row%d' % len(ids))))
    return np.array(X), tag_lists, ids


def _extract_from_csv(path):
    """
    Load already-extracted feature vectors. This is what lets samples/ ship a
    runnable dataset without redistributing .osu files: the CSV holds derived
    statistics and tags, never beatmap content.

    Column contract: beatmap_id, tags (semicolon-joined), then the 90 feature
    columns named by neural_model.FEATURE_NAMES, in that order.
    """
    import pandas as pd
    from neural_model import FEATURE_NAMES

    df = pd.read_csv(path)
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            "%s is missing %d feature column(s), first few: %s"
            % (path, len(missing), missing[:5]))

    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    tag_lists = [[t for t in str(s).split(';') if t] for s in df['tags']]
    ids = [str(i) for i in df['beatmap_id']]
    return X, tag_lists, ids


def prepare_dataset(path='ml_dataset.json', use_cache=True, cache_dir=CACHE_DIR):
    """
    Dataset file -> features, labels, ids. Cached on the dataset content hash, so
    re-running evaluate or promote does not pay the ~45s extraction again. The
    cache key is the file sha256, which means an edited dataset can never
    silently reuse stale features.
    """
    from sklearn.preprocessing import MultiLabelBinarizer

    if not os.path.exists(path):
        raise FileNotFoundError("Dataset not found: %s" % path)

    sha = dataset_sha256(path)
    cache_path = os.path.join(cache_dir, 'features_%s.npz' % sha[:16])

    X = tag_lists = ids = None
    if use_cache and os.path.exists(cache_path):
        blob = np.load(cache_path, allow_pickle=False)
        X = blob['X']
        ids = [str(i) for i in blob['ids']]
        tag_lists = json.loads(str(blob['tags_json']))
        print("[split] Loaded %d cached feature rows from %s" % (X.shape[0], cache_path))

    if X is None:
        print("[split] Extracting features from %s (no cache for %s)..." % (path, sha[:16]))
        if path.lower().endswith('.csv'):
            X, tag_lists, ids = _extract_from_csv(path)
        else:
            X, tag_lists, ids = _extract_from_json(path)
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(
                cache_path, X=X, ids=np.array(ids),
                tags_json=np.array(json.dumps(tag_lists)))
            print("[split] Cached features to %s" % cache_path)

    if X.size == 0:
        raise ValueError("No usable rows extracted from %s" % path)

    binarizer = MultiLabelBinarizer()
    y = binarizer.fit_transform(tag_lists)

    return Prepared(
        X=X, y=y, classes=list(binarizer.classes_), ids=ids,
        tag_lists=tag_lists, dataset_sha=sha, source_path=path)


def fixed_split(prepared, seed=SPLIT_SEED, test_size=TEST_SIZE):
    """
    The evaluation split. Splits INDICES rather than the arrays themselves, which
    is equivalent to the original call (train_test_split shuffles the same way
    regardless of what it is slicing) but lets us recover which beatmap landed in
    the holdout.
    """
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(prepared.X))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)

    holdout_ids = sorted(prepared.ids[i] for i in test_idx)
    split_hash = hashlib.sha256(','.join(holdout_ids).encode()).hexdigest()

    return Split(train_idx=train_idx, test_idx=test_idx, holdout_ids=holdout_ids,
                 split_hash=split_hash, seed=seed, test_size=test_size)


def write_split_manifest(prepared, split, path=MANIFEST_PATH):
    """Record the split so a later run can prove it evaluated the same maps."""
    manifest = {
        'seed': split.seed,
        'test_size': split.test_size,
        'dataset_path': prepared.source_path,
        'dataset_sha256': prepared.dataset_sha,
        'n_rows': int(len(prepared.X)),
        'n_features': int(prepared.X.shape[1]),
        'n_labels': int(len(prepared.classes)),
        'n_holdout': int(len(split.holdout_ids)),
        'holdout_id_sha256': split.split_hash,
        'holdout_beatmap_ids': split.holdout_ids,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def scale_all(prepared):
    """
    Fit StandardScaler on every row, exactly as the original pipeline does. See
    the module docstring: this leaks, it is preserved on purpose, and it is
    identical for every model the gate compares.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(prepared.X), scaler


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='Print and record the fixed evaluation split.')
    ap.add_argument('--dataset', default='ml_dataset.json')
    ap.add_argument('--manifest', default=MANIFEST_PATH)
    ap.add_argument('--no-cache', action='store_true')
    a = ap.parse_args()

    prep = prepare_dataset(a.dataset, use_cache=not a.no_cache)
    sp = fixed_split(prep)
    m = write_split_manifest(prep, sp, a.manifest)
    print(json.dumps({k: v for k, v in m.items() if k != 'holdout_beatmap_ids'}, indent=2))
