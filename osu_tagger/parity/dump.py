"""
Emit the map feature vector for one .osu file as JSON (was parity_dump.py).

Uses the REAL production extractor (osu_tagger.parsing.OsuFileParser +
features.v1.ImprovedBeatmapClassifier for v1, features.v2 for v2), so the
output is exactly what training sees. Its C# counterpart is
OsuScoutNew/parity/ParityDump. Compare the two dumps with
OsuScoutNew/parity/compare_parity.py to prove the feature math agrees.

Usage:
    python -m osu_tagger.parity.dump [--feature-version {1,2}] <path-to.osu> [out.json]

Prints the JSON to stdout, and also writes it to out.json if given. The default
is v1, the vector the shipped app computes, so the app repo's parity harness
keeps working unchanged.
"""

import sys
import json

import numpy as np

from osu_tagger.parsing import OsuFileParser


def dump(osu_path: str, feature_version: int = 1):
    if feature_version == 2:
        from osu_tagger.features.v2 import FEATURE_NAMES_V2, extract_features_v2_from_osu

        vector = extract_features_v2_from_osu(osu_path)
        names = FEATURE_NAMES_V2
    else:
        from osu_tagger.features.v1 import FEATURE_NAMES, ImprovedBeatmapClassifier

        parser = OsuFileParser(osu_path)
        parser.read_file()
        hit_objects = parser.extract_raw_hit_objects()

        clf = ImprovedBeatmapClassifier()
        sections = clf.split_beatmap_into_sections(hit_objects)
        vector = clf._aggregate_features_for_map(sections)
        names = FEATURE_NAMES

    if vector is None:
        raise SystemExit(
            "ERROR: no sections/features produced (map too short or unparseable)"
        )

    features = [float(x) for x in np.asarray(vector).ravel()]
    payload = {
        "source": "python",
        "file": osu_path,
        "length": len(features),
        "features": features,
    }
    if feature_version == 2:
        # v2 carries its names so the C# side can report which feature differs
        # rather than which index.
        payload["feature_version"] = 2
        payload["feature_names"] = list(names)
    return payload


def main():
    args = sys.argv[1:]
    feature_version = 1
    if args[:1] == ["--feature-version"]:
        if len(args) < 2 or args[1] not in ("1", "2"):
            print("--feature-version must be 1 or 2", file=sys.stderr)
            raise SystemExit(2)
        feature_version = int(args[1])
        args = args[2:]

    if not args:
        print("usage: python -m osu_tagger.parity.dump [--feature-version {1,2}] <path-to.osu> [out.json]",
              file=sys.stderr)
        raise SystemExit(2)

    payload = dump(args[0], feature_version)
    text = json.dumps(payload, indent=2)

    if len(args) >= 2:
        with open(args[1], "w", encoding="utf-8") as f:
            f.write(text)

    print(text)


if __name__ == "__main__":
    main()
