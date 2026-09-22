"""
Tests for the promotion gate.

promote.decide is pure - measurements in, a verdict out - so these tests need no
models, no dataset and no mlflow. That is deliberate: the gate is the one piece
of this pipeline whose logic errors are silent (a gate that always passes looks
exactly like a gate that works), so it gets the cheapest, fastest tests in the
repo and they run on every CI push.

The constants here are the measured ones from VERIFIED.md, so a test that fails
after a recalibration is telling you something real rather than being brittle.
"""
import pytest

from mlops.promote import (
    DEFAULT_TOLERANCE,
    MICRO_F1_SIGMA,
    NEVER_PREDICTED_ALLOWANCE,
    REFERENCE_MICRO_F1,
    TOLERANCE_K,
    Scores,
    Verdict,
    decide,
    summarise_spread,
)

# The v1 champion, and a tolerance wide enough to cover ordinary training noise.
CHAMPION = Scores(micro_f1=0.556967, never_predicted_included=1)
TOL = DEFAULT_TOLERANCE


def cand(micro, never=1):
    return Scores(micro_f1=micro, never_predicted_included=never)


class TestCalibratedConstants:
    """
    Pin the measured values. If someone changes a constant without redoing the
    calibration, this is where it surfaces.
    """

    def test_tolerance_is_k_times_sigma(self):
        assert DEFAULT_TOLERANCE == pytest.approx(TOLERANCE_K * MICRO_F1_SIGMA)

    def test_k_is_four_not_two_or_three(self):
        """
        k=4 is not a rounding of 'about 3 sigma'. The champion sits ~1.9 sigma
        above the seed mean, so accepting a candidate at mean-2sigma needs
        (champion-mean) + 2 sigma ~= 3.9 sigma. k=3 rejects honest reruns.
        """
        assert TOLERANCE_K == 4

    def test_reference_is_a_fixed_anchor_not_a_best_ever(self):
        """
        The ratchet floor must not be the maximum of past runs: that is biased
        upward and only ever rises, so the gate would tighten on its own.
        """
        assert REFERENCE_MICRO_F1 == pytest.approx(0.5569668976135489)


class TestMicroF1Checks:
    def test_no_champion_promotes_when_reference_is_cleared(self):
        """Something has to go first - but it still may not be junk."""
        v = decide(cand(0.5550), champion=None)
        assert v.promote is True
        assert 'no champion' in v.reason

    def test_no_champion_still_rejects_a_bad_model(self):
        """
        An empty registry is not a licence to ship anything. The fixed reference
        applies even on the very first promotion.
        """
        v = decide(cand(0.4140), champion=None)
        assert v.promote is False
        assert 'reference' in v.reason

    def test_clearly_better_promotes(self):
        v = decide(cand(0.5600), CHAMPION)
        assert v.promote is True

    def test_worse_within_tolerance_promotes(self):
        """Slightly worse is allowed, or noise alone rejects honest reruns."""
        v = decide(cand(CHAMPION.micro_f1 - TOL / 2), CHAMPION)
        assert v.promote is True
        assert 'within tolerance' in v.reason

    def test_worse_beyond_tolerance_rejects(self):
        v = decide(cand(CHAMPION.micro_f1 - TOL * 2), CHAMPION)
        assert v.promote is False
        assert 'micro_f1 vs champion' in v.reason

    def test_boundary_is_inclusive(self):
        """
        candidate == champion - tolerance passes. Pinned because an off-by-one
        flips every borderline candidate and nothing else would reveal it.
        """
        assert decide(cand(CHAMPION.micro_f1 - TOL), CHAMPION).promote is True

    def test_just_below_boundary_rejects(self):
        assert decide(cand(CHAMPION.micro_f1 - TOL - 1e-6), CHAMPION).promote is False


class TestRatchetGuard:
    """
    The failure the fixed reference exists to prevent: each promotion resets the
    champion, so a chain of within-tolerance candidates walks quality downhill
    while every individual step looks fine.
    """

    def test_candidate_beating_a_degraded_champion_still_fails_the_reference(self):
        degraded = Scores(micro_f1=REFERENCE_MICRO_F1 - 0.03, never_predicted_included=1)
        candidate = cand(degraded.micro_f1 + 0.001)       # better than the incumbent
        v = decide(candidate, degraded)
        assert v.promote is False
        assert 'reference' in v.reason

    def test_reference_can_be_disabled_explicitly(self):
        degraded = Scores(micro_f1=REFERENCE_MICRO_F1 - 0.03, never_predicted_included=1)
        v = decide(cand(degraded.micro_f1 + 0.001), degraded, reference_micro_f1=None)
        assert v.promote is True

    def test_a_simulated_chain_cannot_drift_below_the_reference(self):
        """
        Ten successive promotions, each as bad as the gate permits. Without a
        fixed floor this walks down by 10x tolerance; with one it cannot.
        """
        champion = CHAMPION
        for _ in range(10):
            candidate = cand(champion.micro_f1 - TOL)
            v = decide(candidate, champion)
            if v.promote:
                champion = candidate
        assert champion.micro_f1 >= REFERENCE_MICRO_F1 - TOL


class TestNeverPredictedGuard:
    def test_going_mute_on_extra_supported_tags_rejects(self):
        v = decide(cand(0.5600, never=CHAMPION.never_predicted_included + 5), CHAMPION)
        assert v.promote is False
        assert 'never-predicted' in v.reason

    def test_allowance_admits_ordinary_variation(self):
        """
        Measured across 10 identical reruns this count was 1 or 2 against the
        champion's 1, so +1 must pass.
        """
        v = decide(cand(0.5550, never=CHAMPION.never_predicted_included
                        + NEVER_PREDICTED_ALLOWANCE), CHAMPION)
        assert v.promote is True

    def test_one_beyond_the_allowance_rejects(self):
        v = decide(cand(0.5550, never=CHAMPION.never_predicted_included
                        + NEVER_PREDICTED_ALLOWANCE + 1), CHAMPION)
        assert v.promote is False

    def test_skipped_when_not_measurable(self):
        v = decide(Scores(micro_f1=0.5550), Scores(micro_f1=CHAMPION.micro_f1))
        assert v.promote is True
        assert any('skipped' in d for _, _, d in v.checks)


class TestVerdictReporting:
    def test_every_check_is_recorded_even_on_a_pass(self):
        """A pass should be as auditable as a failure."""
        v = decide(cand(0.5600), CHAMPION)
        names = [n for n, _, _ in v.checks]
        assert names == ['micro_f1 vs champion', 'micro_f1 vs reference',
                         'never-predicted tags']

    def test_report_marks_the_failing_check(self):
        v = decide(cand(0.4000), CHAMPION)
        assert '[FAIL]' in v.report()
        assert v.report().startswith('REJECT:')

    def test_stringifies_for_logs(self):
        assert str(decide(cand(0.5600), CHAMPION)).startswith('PROMOTE:')
        assert str(Verdict(promote=False, reason='nope')).startswith('REJECT:')


class TestToleranceValidation:
    def test_negative_tolerance_is_rejected(self):
        with pytest.raises(ValueError, match='Tolerance must be'):
            decide(cand(0.5550), CHAMPION, tolerance=-0.01)

    def test_zero_tolerance_is_strict_but_legal(self):
        assert decide(cand(CHAMPION.micro_f1), CHAMPION, tolerance=0.0).promote is True
        assert decide(cand(CHAMPION.micro_f1 - 1e-9), CHAMPION,
                      tolerance=0.0).promote is False


class TestSummariseSpread:
    def test_reports_both_stdev_and_gap(self):
        """
        Both, because the tolerance is derived from stdev while the gap is the
        thing that misled the first calibration - keeping both visible is what
        makes that mistake hard to repeat.
        """
        s = summarise_spread([0.50, 0.52, 0.51])
        assert s['stdev'] == pytest.approx(0.01)
        assert s['max_pairwise_gap'] == pytest.approx(0.02)
        assert s['gap_over_stdev'] == pytest.approx(2.0)

    def test_identical_runs_have_zero_spread(self):
        s = summarise_spread([0.5, 0.5])
        assert s['max_pairwise_gap'] == 0.0
        assert s['stdev'] == 0.0
        assert s['gap_over_stdev'] == 0.0

    def test_single_run_is_not_a_spread(self):
        with pytest.raises(ValueError, match='at least 2'):
            summarise_spread([0.5])
