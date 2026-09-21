"""
Tests for the feature extractor.

These exist because of a specific risk: the 90-feature vector is computed twice,
once here in Python for training and once in C# (FeatureExtractor.cs) for
inference in the desktop app. If the Python side changes and the C# side does
not, nothing crashes - the app just standardises the features with constants
that no longer describe them and quietly returns wrong tags for every map.

So the golden-vector test below is the important one. It pins the exact 90
floats produced for one committed beatmap. Any edit that changes the math fails
it immediately, which is the prompt to either revert or make the matching C#
change and regenerate the parity goldens.
"""
import json
import os

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'golden_feature_vector.json')


@pytest.fixture(scope='module')
def classifier():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    from neural_model import ImprovedBeatmapClassifier
    return ImprovedBeatmapClassifier()


@pytest.fixture(scope='module')
def golden():
    with open(GOLDEN_PATH, encoding='utf-8') as f:
        return json.load(f)


class TestConstants:
    def test_feature_names_match_feature_count(self):
        from neural_model import BASE_FEATURE_NAMES, FEATURE_COUNT
        assert len(BASE_FEATURE_NAMES) == FEATURE_COUNT

    def test_aggregated_vector_is_ninety_features(self):
        """
        90 = 29 base features x (max, mean, std) + 3 hybrid flags. The C# app
        validates this exact length, so it is a contract, not a coincidence.
        """
        from neural_model import AGGREGATED_FEATURE_COUNT, FEATURE_COUNT
        assert FEATURE_COUNT == 29
        assert AGGREGATED_FEATURE_COUNT == 90
        assert AGGREGATED_FEATURE_COUNT == FEATURE_COUNT * 3 + 3

    def test_feature_names_are_unique(self):
        from neural_model import FEATURE_NAMES
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)

    def test_threshold_is_the_shipped_value(self):
        from ensemble_evaluator import THRESHOLD
        assert THRESHOLD == 0.27


class TestExtractMeaningfulFeatures:
    def test_empty_input_returns_zero_vector(self, classifier):
        from neural_model import FEATURE_COUNT
        out = classifier.extract_meaningful_features([])
        assert out.shape == (FEATURE_COUNT,)
        assert np.all(out == 0)

    def test_too_few_objects_returns_zero_vector(self, classifier):
        """Below MIN_OBJECTS_FOR_FEATURES the statistics are meaningless."""
        from neural_model import MIN_OBJECTS_FOR_FEATURES
        objs = [[100, 100, t * 100, 1, None, [], 1, 0]
                for t in range(MIN_OBJECTS_FOR_FEATURES - 1)]
        out = classifier.extract_meaningful_features(objs)
        assert np.all(out == 0)

    def test_zero_duration_returns_zero_vector(self, classifier):
        """All objects at the same timestamp would divide by zero."""
        objs = [[100, 100, 500, 1, None, [], 1, 0] for _ in range(10)]
        out = classifier.extract_meaningful_features(objs)
        assert np.all(out == 0)

    def test_returns_finite_values_for_a_normal_section(self, classifier):
        from neural_model import FEATURE_COUNT
        objs = [[100 + i * 30, 200, i * 120, 1, None, [], 1, 0] for i in range(40)]
        out = classifier.extract_meaningful_features(objs)
        assert out.shape == (FEATURE_COUNT,)
        assert np.all(np.isfinite(out)), "NaN/inf would poison the scaler silently"

    def test_detects_a_stream(self, classifier):
        """
        Tightly spaced, closely placed objects are a stream by definition
        (gap < STREAM_GAP_MS and spacing < STREAM_MAX_SPACING_PX).
        """
        from neural_model import STREAM_MIN_LENGTH
        objs = [[100 + i * 10, 200, i * 100, 1, None, [], 1, 0] for i in range(20)]
        out = classifier.extract_meaningful_features(objs)
        assert out[2] >= STREAM_MIN_LENGTH, "max_continuous_stream should see the stream"

    def test_far_apart_objects_are_not_a_stream(self, classifier):
        """Cross-screen jumps at high BPM must not count as a stream."""
        objs = [[50 + (i % 2) * 400, 200, i * 100, 1, None, [], 1, 0] for i in range(20)]
        out = classifier.extract_meaningful_features(objs)
        assert out[2] == 0


class TestSectionSplitting:
    def test_gap_longer_than_threshold_splits(self, classifier):
        from neural_model import BREAK_THRESHOLD_MS, MIN_SECTION_LENGTH
        first = [[100, 100, i * 100, 1, None, [], 1, 0] for i in range(MIN_SECTION_LENGTH + 5)]
        start = first[-1][2] + BREAK_THRESHOLD_MS + 1
        second = [[100, 100, start + i * 100, 1, None, [], 1, 0]
                  for i in range(MIN_SECTION_LENGTH + 5)]
        assert len(classifier.split_beatmap_into_sections(first + second)) == 2

    def test_gap_at_threshold_does_not_split(self, classifier):
        """The split is on `> threshold`, so exactly the threshold stays joined."""
        from neural_model import BREAK_THRESHOLD_MS, MIN_SECTION_LENGTH
        first = [[100, 100, i * 100, 1, None, [], 1, 0] for i in range(MIN_SECTION_LENGTH + 5)]
        start = first[-1][2] + BREAK_THRESHOLD_MS
        second = [[100, 100, start + i * 100, 1, None, [], 1, 0]
                  for i in range(MIN_SECTION_LENGTH + 5)]
        assert len(classifier.split_beatmap_into_sections(first + second)) == 1

    def test_short_sections_are_dropped(self, classifier):
        from neural_model import BREAK_THRESHOLD_MS, MIN_SECTION_LENGTH
        long_part = [[100, 100, i * 100, 1, None, [], 1, 0]
                     for i in range(MIN_SECTION_LENGTH + 5)]
        start = long_part[-1][2] + BREAK_THRESHOLD_MS + 1
        stub = [[100, 100, start + i * 100, 1, None, [], 1, 0] for i in range(3)]
        assert len(classifier.split_beatmap_into_sections(long_part + stub)) == 1

    def test_no_objects_gives_no_sections(self, classifier):
        assert classifier.split_beatmap_into_sections([]) == []

    def test_unbroken_map_is_one_section(self, classifier):
        from neural_model import MIN_SECTION_LENGTH
        objs = [[100, 100, i * 100, 1, None, [], 1, 0] for i in range(MIN_SECTION_LENGTH + 10)]
        assert len(classifier.split_beatmap_into_sections(objs)) == 1


class TestAggregation:
    def test_aggregated_vector_length(self, classifier):
        from neural_model import AGGREGATED_FEATURE_COUNT
        objs = [[100 + i * 20, 150, i * 110, 1, None, [], 1, 0] for i in range(40)]
        vec = classifier._aggregate_features_for_map([objs])
        assert vec.shape == (AGGREGATED_FEATURE_COUNT,)

    def test_hybrid_flags_are_binary(self, classifier):
        objs = [[100 + i * 20, 150, i * 110, 1, None, [], 1, 0] for i in range(40)]
        vec = classifier._aggregate_features_for_map([objs])
        assert set(vec[-3:]).issubset({0.0, 1.0})

    def test_no_sections_returns_none(self, classifier):
        assert classifier._aggregate_features_for_map([]) is None


class TestGoldenVector:
    """
    The parity regression. See this module's docstring for why it matters.

    If this fails, the feature math changed. That is only correct if you also
    changed FeatureExtractor.cs in the app repo and regenerated its goldens - so
    the fix is never "update the golden until it passes".
    """

    def test_golden_file_describes_a_committed_map(self, golden):
        assert os.path.exists(os.path.join(REPO_ROOT, golden['osu_file'])), \
            "golden references a map that is not in the repo"

    def test_golden_vector_is_unchanged(self, classifier, golden):
        from osu_parser import OsuFileParser

        parser = OsuFileParser(os.path.join(REPO_ROOT, golden['osu_file']))
        parser.read_file()
        sections = classifier.split_beatmap_into_sections(parser.extract_raw_hit_objects())
        vec = classifier._aggregate_features_for_map(sections)

        assert vec is not None
        assert len(vec) == len(golden['features'])
        np.testing.assert_allclose(
            vec, np.array(golden['features']), rtol=1e-9, atol=1e-9,
            err_msg="Feature math changed. If deliberate, mirror it in the C# "
                    "FeatureExtractor and regenerate both sets of goldens.")

    def test_golden_section_count_is_unchanged(self, classifier, golden):
        from osu_parser import OsuFileParser

        parser = OsuFileParser(os.path.join(REPO_ROOT, golden['osu_file']))
        parser.read_file()
        sections = classifier.split_beatmap_into_sections(parser.extract_raw_hit_objects())
        assert len(sections) == golden['n_sections']
