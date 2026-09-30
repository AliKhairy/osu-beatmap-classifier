"""
Batch drift report: does a folder of new maps look like what the model was
trained on?

The question this answers is narrow and worth stating precisely. It compares
FEATURE DISTRIBUTIONS - the 90 numbers the model actually consumes - between the
training set and a folder of .osu files. It does not measure accuracy, because
new maps have no tags to score against. Drift here is a prompt to look, not a
verdict that the model has got worse.

Interpreting the number needs one caveat kept in view: a folder of 31 maps is a
small sample, so some columns will read as drifted from sampling noise alone.
The useful signal is which features drift and whether the set is stable across
folders, not the raw count.
"""
import os

import numpy as np

REPORT_NAME = 'drift_report.html'


def features_from_folder(folder, max_maps=None):
    """
    Run the production extractor over every .osu file in a folder.

    Uses the same OsuFileParser and _aggregate_features_for_map as training, so
    the two sides of the comparison are produced by identical code. Files that
    fail to parse or are too short are skipped and counted, not silently
    dropped - a folder where most maps fail should look like a problem, not like
    a clean report over three maps.
    """
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    from neural_model import ImprovedBeatmapClassifier
    from osu_parser import OsuFileParser

    if not os.path.isdir(folder):
        raise NotADirectoryError("Not a folder: %s" % folder)

    osu_files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.osu'))
    if max_maps:
        osu_files = osu_files[:max_maps]
    if not osu_files:
        raise ValueError("No .osu files found in %s" % folder)

    classifier = ImprovedBeatmapClassifier()
    rows, names, skipped = [], [], []

    for filename in osu_files:
        path = os.path.join(folder, filename)
        try:
            parser = OsuFileParser(path)
            parser.read_file()
            sections = classifier.split_beatmap_into_sections(parser.extract_raw_hit_objects())
            if not sections:
                skipped.append((filename, 'no playable sections'))
                continue
            vec = classifier._aggregate_features_for_map(sections)
            if vec is None:
                skipped.append((filename, 'feature extraction returned None'))
                continue
            rows.append(vec)
            names.append(filename)
        except Exception as e:                      # noqa: BLE001 - report, do not crash
            skipped.append((filename, str(e)))

    if not rows:
        raise ValueError(
            "Extracted features from 0 of %d .osu files in %s" % (len(osu_files), folder))

    return np.array(rows), names, skipped


def build_report(reference_X, current_X, out_path=REPORT_NAME):
    """
    Evidently data-drift report over the 90 named feature columns.

    Returns (summary dict, path). The summary carries the share of drifted
    columns, which is the single number worth logging as a metric.
    """
    import pandas as pd
    from evidently import Dataset, DataDefinition, Report
    from evidently.presets import DataDriftPreset

    from neural_model import FEATURE_NAMES

    ref_df = pd.DataFrame(reference_X, columns=FEATURE_NAMES)
    cur_df = pd.DataFrame(current_X, columns=FEATURE_NAMES)

    definition = DataDefinition(numerical_columns=list(FEATURE_NAMES))
    ref_ds = Dataset.from_pandas(ref_df, data_definition=definition)
    cur_ds = Dataset.from_pandas(cur_df, data_definition=definition)

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(current_data=cur_ds, reference_data=ref_ds)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    result.save_html(out_path)

    summary = _extract_drift_summary(result)
    summary.update({
        'n_reference': int(reference_X.shape[0]),
        'n_current': int(current_X.shape[0]),
        'n_features': int(reference_X.shape[1]),
        'report_path': out_path,
    })
    return summary, out_path


DRIFTED_COUNT_TYPE = 'evidently:metric_v2:DriftedColumnsCount'
VALUE_DRIFT_TYPE = 'evidently:metric_v2:ValueDrift'


def _extract_drift_summary(result, top_n=8):
    """
    Pull the drift counts out of Evidently's result.

    Targets the DriftedColumnsCount metric by its config type and reads only its
    `value` block. That specificity is deliberate and was learned the hard way: a
    generic recursive search for a key named "drift_share" finds
    `config.drift_share`, which is the THRESHOLD Evidently was configured with
    (0.5), not the measured share. The first version of this function did
    exactly that and cheerfully reported 0.5 for a dataset where 88 of 90
    columns had drifted.

    Structure being parsed (evidently 0.7.23):
        {"metrics": [
            {"config": {"type": "...DriftedColumnsCount", "drift_share": 0.5},
             "value": {"count": 88.0, "share": 0.978}},        <- the measurement
            {"config": {"type": "...ValueDrift", "column": "max_burst_count",
                        "threshold": 0.1},
             "value": 0.529},                                   <- per column
            ...]}

    If the expected shape is absent, every field stays None and
    drift_summary_parsed is False. A monitoring tool that reports a confident
    wrong number is worse than one that admits it could not read the result.
    """
    out = {'drifted_columns': None, 'drift_share': None,
           'drift_threshold': None, 'top_drifted': None,
           'drift_summary_parsed': False}
    try:
        payload = result.dict()
    except Exception:
        return out

    metrics = payload.get('metrics')
    if not isinstance(metrics, list):
        return out

    per_column = []
    for entry in metrics:
        if not isinstance(entry, dict):
            continue
        config = entry.get('config') or {}
        etype = config.get('type')
        value = entry.get('value')

        if etype == DRIFTED_COUNT_TYPE and isinstance(value, dict):
            count, share = value.get('count'), value.get('share')
            if isinstance(count, (int, float)):
                out['drifted_columns'] = int(count)
            if isinstance(share, (int, float)):
                out['drift_share'] = float(share)
            # Recorded separately so it can never be mistaken for the result.
            if isinstance(config.get('drift_share'), (int, float)):
                out['drift_threshold'] = float(config['drift_share'])
            out['drift_summary_parsed'] = out['drifted_columns'] is not None

        elif etype == VALUE_DRIFT_TYPE and isinstance(value, (int, float)):
            column = config.get('column')
            if column:
                per_column.append((str(column), float(value)))

    if per_column:
        per_column.sort(key=lambda kv: kv[1], reverse=True)
        out['top_drifted'] = per_column[:top_n]

    return out
