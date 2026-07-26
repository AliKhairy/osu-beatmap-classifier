"""
parity_dump.py — emit the aggregated map feature vector for one .osu file as JSON.

Uses the REAL production extractor (osu_parser.OsuFileParser +
neural_model.ImprovedBeatmapClassifier), so the output is exactly what training
sees. Its C# counterpart is OsuScoutNew/parity/ParityDump. Compare the two dumps
with OsuScoutNew/parity/compare_parity.py to prove the feature math agrees.

Usage:
    python parity_dump.py <path-to.osu> [out.json]

Prints the JSON to stdout, and also writes it to out.json if given.
"""

import sys
import json

import numpy as np

from osu_parser import OsuFileParser
from neural_model import ImprovedBeatmapClassifier


def dump(osu_path: str):
    parser = OsuFileParser(osu_path)
    parser.read_file()
    hit_objects = parser.extract_raw_hit_objects()

    clf = ImprovedBeatmapClassifier()
    sections = clf.split_beatmap_into_sections(hit_objects)
    vector = clf._aggregate_features_for_map(sections)

    if vector is None:
        raise SystemExit(
            "ERROR: no sections/features produced (map too short or unparseable)"
        )

    features = [float(x) for x in np.asarray(vector).ravel()]
    return {
        "source": "python",
        "file": osu_path,
        "length": len(features),
        "features": features,
    }


def main():
    if len(sys.argv) < 2:
        print("usage: python parity_dump.py <path-to.osu> [out.json]", file=sys.stderr)
        raise SystemExit(2)

    payload = dump(sys.argv[1])
    text = json.dumps(payload, indent=2)

    if len(sys.argv) >= 3:
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            f.write(text)

    print(text)


if __name__ == "__main__":
    main()
