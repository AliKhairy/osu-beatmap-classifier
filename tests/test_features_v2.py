"""
Tests for the v2 feature extractor.

Each synthetic map below reproduces one specific way v1 mis-measures a pattern,
and asserts v2 measures it the way the tag it feeds is defined. They are the
executable form of the v2 spec, and what the C# port has to agree with.

Maps are built at 200 BPM (beat 300 ms), where v1's fixed 165 ms stream cutoff
already swallows 1/2 rhythm, so v1's failure and v2's fix show up on the same
input.
"""
import json
import math
import os

import numpy as np
import pytest

from osu_tagger.features.v2 import (
    DEATHSTREAM_MIN_NOTES,
    FEATURE_COUNT_V2,
    FEATURE_NAMES_V2,
    _Timing,
    circle_radius,
    extract_features_v2,
    extract_features_v2_from_osu,
    slider_path_end,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_V2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'golden_feature_vector_v2.json')

BEAT = 300.0                                   # 200 BPM
TIMING = [(0.0, BEAT, True)]
DIFF = {'CircleSize': 4.0, 'ApproachRate': 9.0, 'OverallDifficulty': 8.0,
        'SliderMultiplier': 1.4}


def circle(x, y, t):
    return [x, y, int(t), 1, None, [], 1, 0.0]


def feats(objs, difficulty=DIFF, timing=TIMING):
    v = extract_features_v2(objs, difficulty, timing)
    return dict(zip(FEATURE_NAMES_V2, v))


def back_and_forth(n=40, spacing=100, step=BEAT / 2):
    """Jumps between two points at 1/2: the 1-2 pattern."""
    return [circle(150 + (i % 2) * spacing, 200, i * step) for i in range(n)]


class TestShape:
    def test_names_match_the_vector(self):
        assert len(FEATURE_NAMES_V2) == FEATURE_COUNT_V2
        assert len(extract_features_v2(back_and_forth(), DIFF, TIMING)) == FEATURE_COUNT_V2

    def test_finite_for_a_normal_map(self):
        v = extract_features_v2(back_and_forth(), DIFF, TIMING)
        assert np.all(np.isfinite(v)), "NaN/inf would poison the scaler silently"

    @pytest.mark.parametrize('objs', [
        [],
        [circle(100, 100, i * 150) for i in range(4)],        # below MIN_OBJECTS
        [circle(100, 100, 500) for _ in range(10)],            # zero duration
    ])
    def test_unmeasurable_maps_are_all_zero(self, objs):
        assert np.all(extract_features_v2(objs, DIFF, TIMING) == 0)

    def test_missing_metadata_falls_back_instead_of_failing(self):
        v = extract_features_v2(back_and_forth(), None, None)
        assert np.all(np.isfinite(v))


class TestRhythmIsBpmRelative:
    def test_half_beat_jumps_at_200_bpm_are_not_a_stream(self):
        """
        1/2 at 200 BPM is 150 ms apart - under v1's 165 ms cutoff - so v1 calls
        these back-and-forth jumps a stream. v2 judges rhythm against the beat.
        """
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        from osu_tagger.features.v1 import ImprovedBeatmapClassifier
        objs = back_and_forth(spacing=100)
        v1 = ImprovedBeatmapClassifier().extract_meaningful_features(objs)
        assert v1[2] >= 8, "v1's max_continuous_stream should show the old bug"

        f = feats(objs)
        assert f['log_longest_chain'] == 0
        assert f['streams_per_min'] == f['bursts_per_min'] == 0
        assert f['snap_frac_1_2'] == pytest.approx(1.0)

    def test_triples_are_counted_as_triples_only(self):
        objs, t = [], 0.0
        for g in range(12):
            for k in range(3):
                objs.append(circle(100 + k * 30 + (g % 2) * 200, 200, t + k * BEAT / 4))
            t += BEAT / 4 * 2 + BEAT / 2            # 1/2 gap after each group
        f = feats(objs)
        assert f['triples_per_min'] > 0
        assert f['doubles_per_min'] == f['bursts_per_min'] == f['streams_per_min'] == 0

    def test_doubles_are_counted_as_doubles_only(self):
        objs, t = [], 0.0
        for g in range(12):
            for k in range(2):
                objs.append(circle(100 + k * 30 + (g % 2) * 200, 200, t + k * BEAT / 4))
            t += BEAT / 4 + BEAT / 2
        f = feats(objs)
        assert f['doubles_per_min'] > 0
        assert f['triples_per_min'] == f['bursts_per_min'] == 0

    @pytest.mark.parametrize('notes,bucket', [
        (DEATHSTREAM_MIN_NOTES - 1, 'streams_per_min'),
        (DEATHSTREAM_MIN_NOTES, 'deathstreams_per_min'),
    ])
    def test_deathstream_boundary_follows_echosu(self, notes, bucket):
        """echosu: a deathstream is 'more than 60 notes'."""
        objs = [circle(100 + (i % 10) * 20, 200, i * BEAT / 4) for i in range(notes)]
        f = feats(objs)
        assert f[bucket] > 0
        other = 'deathstreams_per_min' if bucket == 'streams_per_min' else 'streams_per_min'
        assert f[other] == 0

    def test_third_snaps_are_their_own_class(self):
        objs = [circle(100 + (i % 2) * 100, 200, i * BEAT / 3) for i in range(30)]
        assert feats(objs)['snap_frac_1_3'] == pytest.approx(1.0)


class TestTimeProfile:
    def test_a_perfectly_regular_map_has_no_trend(self):
        """
        Every window has the same load, so any 'trend' would be rounding noise -
        which Python and C# sum in different orders and would disagree on.
        """
        # 250 ms apart: exactly 16 notes in every 4 s window, so both loads are
        # constant (the aim load up to rounding).
        f = feats(back_and_forth(n=400, step=250))
        assert f['aim_trend'] == 0 and f['speed_trend'] == 0


class TestSpinnersAreNotNotes:
    def test_a_spinner_changes_nothing(self):
        """v1 treated a spinner as a note at (256, 192), adding fake jumps."""
        objs = back_and_forth()
        with_spinner = objs[:20] + [[256, 192, int(objs[19][2] + 60), 8, None, [], 1, 0.0]] + objs[20:]
        np.testing.assert_allclose(extract_features_v2(with_spinner, DIFF, TIMING),
                                   extract_features_v2(objs, DIFF, TIMING))


class TestSliderEnds:
    def test_linear_path_end(self):
        np.testing.assert_allclose(slider_path_end((100, 100), 'L', [[300, 100]], 150), [250, 100])

    def test_path_is_extended_past_its_last_point(self):
        np.testing.assert_allclose(slider_path_end((100, 100), 'L', [[200, 100]], 150), [250, 100])

    def test_perfect_arc_end(self):
        """A semicircle through (0,0), (50,50), (100,0): half its length is the top."""
        r = 50.0
        np.testing.assert_allclose(slider_path_end((0, 0), 'P', [[50, 50], [100, 0]], math.pi * r),
                                   [100, 0], atol=1e-6)
        np.testing.assert_allclose(slider_path_end((0, 0), 'P', [[50, 50], [100, 0]], math.pi * r / 2),
                                   [50, 50], atol=1e-6)

    def test_bezier_walks_its_control_points(self):
        end = slider_path_end((0, 0), 'B', [[100, 0], [100, 100]], 150)
        np.testing.assert_allclose(end, [100, 50])

    def _slider_then_circle(self, slides):
        """A 100 px slider, then a circle placed exactly where an odd slider ends."""
        difficulty = dict(DIFF, SliderMultiplier=2.8)   # span = 100/280 beat, so 2 slides fit in a beat
        objs, t = [], 0.0
        for _ in range(15):
            objs.append([100, 100, int(t), 2, 'L', [[200, 100]], slides, 100.0])
            objs.append(circle(200, 100, t + BEAT))
            t += 2 * BEAT
        return feats(objs, difficulty)

    def test_moves_are_measured_from_where_the_slider_ends(self):
        """
        With one slide the slider ends ON the next circle: no jump. With two it
        ends back at its head, 100 px away: a jump. v1 measured both from the
        head and could not tell them apart.
        """
        odd, even = self._slider_then_circle(1), self._slider_then_circle(2)
        assert even['jump_frac'] > odd['jump_frac']
        assert even['jump_from_slider_frac'] > odd['jump_from_slider_frac'] == 0
        assert even['jump_to_slider_frac'] == odd['jump_to_slider_frac'] > 0


class TestAngles:
    def test_back_and_forth_is_sharp_and_a_1_2(self):
        f = feats(back_and_forth())
        assert f['jump_angle_sharp_frac'] == pytest.approx(1.0)
        assert f['move_reversal_frac'] == pytest.approx(1.0)
        assert f['back_forth_run_frac'] > 0
        assert f['jump_angle_wide_frac'] == 0

    def test_back_and_forth_counts_at_any_rhythm(self):
        """1-2 reversals often sit outside jump rhythm; v1 caught them, v2 must too."""
        f = feats(back_and_forth(step=BEAT / 3))       # 1/3: not a 'jump' by rhythm
        assert f['jump_frac'] == 0
        assert f['move_reversal_frac'] == pytest.approx(1.0)

    def test_one_reversal_is_not_a_back_and_forth_pattern(self):
        objs = [circle(10 + i * 100, 200, i * BEAT / 2) for i in range(5)]
        objs.append(circle(310, 200, 5 * BEAT / 2))     # a single snap back
        f = feats(objs)
        assert f['move_reversal_frac'] > 0
        assert f['back_forth_run_frac'] == 0

    def test_straight_line_jumps_are_wide_and_linear(self):
        objs = [circle(10 + i * 100, 200, i * BEAT / 2) for i in range(6)]
        f = feats(objs)
        assert f['jump_angle_wide_frac'] == pytest.approx(1.0)
        assert f['jump_angle_linear_frac'] == pytest.approx(1.0)
        assert f['jump_angle_sharp_frac'] == 0

    def test_square_pattern(self):
        corners = [(100, 100), (300, 100), (300, 300), (100, 300)]
        objs = [circle(*corners[i % 4], i * BEAT / 2) for i in range(24)]
        f = feats(objs)
        assert f['jump_angle_square_frac'] == pytest.approx(1.0)
        assert f['jump_square_frac'] == pytest.approx(1.0)
        assert f['jump_rotation_consistency'] == pytest.approx(1.0)
        assert f['jump_closed_shape_frac'] > 0

    def test_stream_notes_never_enter_jump_angles(self):
        """v1 counted every note triple, so a zig-zag stream flooded the angle buckets."""
        objs = [circle(150 + (i % 2) * 80, 200 + (i % 3) * 20, i * BEAT / 4) for i in range(40)]
        f = feats(objs)
        assert f['jump_frac'] == 0
        for name in FEATURE_NAMES_V2:
            if name.startswith('jump_angle'):
                assert f[name] == 0, name
        assert f['streams_per_min'] > 0


class TestCircleSize:
    def test_spacing_is_measured_in_radii(self):
        objs = back_and_forth(spacing=200)
        at3 = feats(objs, dict(DIFF, CircleSize=3.0))['jump_dist_p50_radii']
        at6 = feats(objs, dict(DIFF, CircleSize=6.0))['jump_dist_p50_radii']
        assert at3 / at6 == pytest.approx(circle_radius(6.0) / circle_radius(3.0))


class TestTiming:
    def test_red_lines_set_the_beat(self):
        timing = _Timing([(0.0, 300.0, True), (5000.0, 400.0, True)], np.array([0.0]))
        np.testing.assert_allclose(timing.beat_length(np.array([-10.0, 0.0, 4999.0, 5000.0])),
                                   [300, 300, 300, 400])

    def test_green_lines_set_velocity_and_red_lines_reset_it(self):
        timing = _Timing([(0.0, 300.0, True), (1000.0, -50.0, False), (2000.0, 300.0, True)],
                         np.array([0.0]))
        np.testing.assert_allclose(timing.slider_velocity(np.array([500.0, 1500.0, 2500.0])),
                                   [1.0, 2.0, 1.0])

    def test_velocity_is_clamped(self):
        timing = _Timing([(0.0, 300.0, True), (100.0, -1.0, False)], np.array([0.0]))
        assert timing.slider_velocity(np.array([200.0]))[0] == 10.0

    def test_trick_beat_lengths_are_clamped_like_the_game(self):
        """
        Real maps in the dataset use 1e-298 ms red lines. Unclamped, one of them
        made a BPM of 5e301, overflowed the scaler and silently ruined training.
        """
        timing = _Timing([(0.0, 1e-298, True), (1000.0, 1e9, True)], np.array([0.0]))
        np.testing.assert_allclose(timing.beat_length(np.array([500.0, 1500.0])), [6.0, 60000.0])

    def test_trick_timing_keeps_every_feature_bounded(self):
        objs = back_and_forth()
        v = extract_features_v2(objs, DIFF, [(0.0, BEAT, True), (3000.0, 1e-298, True)])
        assert np.all(np.isfinite(v)) and np.abs(v).max() < 1e6


class TestParser:
    def test_difficulty_and_timing_points(self, tmp_path):
        from osu_tagger.parsing import OsuFileParser
        osu = tmp_path / 'map.osu'
        osu.write_text(
            'osu file format v5\n\n[Difficulty]\nCircleSize:4.2\nOverallDifficulty:7\n'
            'SliderMultiplier:1.6\n\n[TimingPoints]\n'
            '1000,333.33,4,2,0,100,1,0\n'
            '2000,-50,4,2,0,100,0,0\n'
            '// a comment\n'
            '3000,-200\n\n'                        # old format: negative means inherited
            '[HitObjects]\n100,100,1000,1,0\n', encoding='utf-8')
        parser = OsuFileParser(str(osu))
        parser.read_file()
        diff = parser.get_difficulty()
        assert diff['CircleSize'] == 4.2
        assert diff['ApproachRate'] == 7.0, "no ApproachRate line: the game uses OD"
        assert parser.get_timing_points() == [(1000.0, 333.33, True), (2000.0, -50.0, False),
                                              (3000.0, -200.0, False)]


class TestGoldenVectorV2:
    """
    Pins the exact v2 vector for one committed map - the same one the v1 golden
    uses. It is the reference the C# port must reproduce. Regenerate only for a
    deliberate change to features_v2, with:

        python -m osu_tagger.parity.dump --feature-version 2 "<osu_file>" tests/golden_feature_vector_v2.json
    """

    def test_golden_vector_is_unchanged(self):
        with open(GOLDEN_V2_PATH, encoding='utf-8') as f:
            golden = json.load(f)
        assert golden['feature_names'] == FEATURE_NAMES_V2
        vec = extract_features_v2_from_osu(os.path.join(REPO_ROOT, golden['file']))
        np.testing.assert_allclose(
            vec, np.array(golden['features']), rtol=1e-9, atol=1e-9,
            err_msg="v2 feature math changed. If deliberate, regenerate this golden.")
