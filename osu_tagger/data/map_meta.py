# osu_tagger/data/map_meta.py
"""
The per-map metadata sidecar that v2 features need and ml_dataset.json lacks.

ml_dataset.json stores each map's hit objects and tags, but not its [Difficulty]
or [TimingPoints]. v2 features need both: circle size to measure spacing in
radii, and beat lengths to judge rhythm as 1/2 or 1/4 instead of a fixed number
of milliseconds.

Rebuilding the dataset to add them would mean re-scraping tags and would move
the holdout. So they live in a separate file, map_meta.json, parsed offline
from the .osu files already sitting in downloads/ and keyed by beatmap id.
ml_dataset.json is never touched, so its hash - and the evaluation split - stay
exactly as they are.
"""
import json
import os

from osu_tagger.parsing import OsuFileParser

META_PATH = 'map_meta.json'
DOWNLOADS_DIR = 'downloads'


def osu_path_for(beatmap_id, downloads_dir=DOWNLOADS_DIR):
    """Where osu_api.get_beatmap_file saved this map."""
    return os.path.join(downloads_dir, 'downloaded_%s.osu' % beatmap_id)


def build_map_meta(dataset_path='ml_dataset.json', downloads_dir=DOWNLOADS_DIR,
                   out_path=META_PATH):
    """
    Parse difficulty and timing points for every map in the dataset.

    Also records how many hit objects the .osu file holds. If the mapper
    updated the map after it was scraped, that count differs from the dataset's
    and the timing may not match the objects; those maps are reported, and the
    report is the prompt to decide whether it matters.

    Returns a summary dict. Writes out_path atomically.
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    maps, missing, mismatched = {}, [], []
    for sample in dataset:
        bid = str(sample.get('beatmap_id'))
        path = osu_path_for(bid, downloads_dir)
        if not os.path.exists(path):
            missing.append(bid)
            continue
        parser = OsuFileParser(path)
        parser.read_file()
        n_objects = len(parser.extract_raw_hit_objects())
        if n_objects != len(sample.get('hit_objects') or []):
            mismatched.append(bid)
        maps[bid] = {
            'difficulty': parser.get_difficulty(),
            'timing_points': [list(p) for p in parser.get_timing_points()],
            'n_objects': n_objects,
        }

    payload = {'dataset_path': dataset_path, 'n_maps': len(maps),
               'missing': missing, 'object_count_mismatch': mismatched, 'maps': maps}
    tmp = out_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, separators=(',', ':'))
    os.replace(tmp, out_path)

    return {'n_dataset': len(dataset), 'n_covered': len(maps),
            'missing': missing, 'object_count_mismatch': mismatched, 'out_path': out_path}


def load_map_meta(path=META_PATH):
    """beatmap_id -> {'difficulty', 'timing_points'}; timing points as tuples."""
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return {bid: {'difficulty': m['difficulty'],
                  'timing_points': [(t, bl, bool(u)) for t, bl, u in m['timing_points']]}
            for bid, m in payload['maps'].items()}
