"""
Tests for the promotion gate.

promote.decide is pure - three floats in, a verdict out - so these tests need no
models, no dataset and no mlflow. That is deliberate: the gate is the one piece
of this pipeline whose logic errors are silent (a gate that always passes looks
exactly like a gate that works), so it gets the cheapest, fastest tests in the
repo and they run on every CI push.
"""
import pytest

from promote import Verdict, decide, summarise_spread

TOL = 0.01


def test_no_champion_promotes_unconditionally():
    """The first model must be able to get in; there is nothing to compare to."""
    v = decide(candidate_f1=0.01, champion_f1=None, best_ever_f1=None, tolerance=TOL)
    assert v.promote is True
    assert 'no champion' in v.reason


def test_clearly_better_promotes():
    v = decide(candidate_f1=0.60, champion_f1=0.50, best_ever_f1=0.50, tolerance=TOL)
    assert v.promote is True


def test_worse_within_tolerance_promotes():
    """Slightly worse is allowed - otherwise noise alone rejects honest reruns."""
    v = decide(candidate_f1=0.495, champion_f1=0.50, best_ever_f1=0.50, tolerance=TOL)
    assert v.promote is True
    assert 'within tolerance' in v.reason


def test_worse_beyond_tolerance_rejects():
    v = decide(candidate_f1=0.40, champion_f1=0.50, best_ever_f1=0.50, tolerance=TOL)
    assert v.promote is False
    assert 'worse than the champion' in v.reason


def test_exactly_at_tolerance_boundary_promotes():
    """
    Boundary is inclusive: candidate == champion - tolerance passes. Pinned
    because an off-by-one here flips the behaviour of every borderline candidate
    and nothing else in the system would reveal it.
    """
    v = decide(candidate_f1=0.49, champion_f1=0.50, best_ever_f1=0.50, tolerance=TOL)
    assert v.promote is True


def test_just_below_tolerance_boundary_rejects():
    v = decide(candidate_f1=0.4899, champion_f1=0.50, best_ever_f1=0.50, tolerance=TOL)
    assert v.promote is False


def test_ratchet_guard_rejects_candidate_that_beats_only_the_incumbent():
    """
    THE regression this gate exists to prevent.

    A candidate better than the current champion but well below the best score
    ever recorded means quality has already drifted down through earlier
    within-tolerance promotions. Accepting it continues the slide, and every
    individual step looks fine.
    """
    v = decide(candidate_f1=0.50, champion_f1=0.49, best_ever_f1=0.60, tolerance=TOL)
    assert v.promote is False
    assert 'best ever' in v.reason


def test_ratchet_guard_allows_candidate_within_tolerance_of_best_ever():
    v = decide(candidate_f1=0.595, champion_f1=0.49, best_ever_f1=0.60, tolerance=TOL)
    assert v.promote is True


def test_best_ever_none_skips_the_ratchet_guard():
    """An empty registry has no history to ratchet away from."""
    v = decide(candidate_f1=0.50, champion_f1=0.49, best_ever_f1=None, tolerance=TOL)
    assert v.promote is True


def test_zero_tolerance_is_strict_but_legal():
    assert decide(0.50, 0.50, 0.50, tolerance=0.0).promote is True
    assert decide(0.4999, 0.50, 0.50, tolerance=0.0).promote is False


def test_negative_tolerance_is_rejected():
    with pytest.raises(ValueError, match='Tolerance must be'):
        decide(0.50, 0.50, 0.50, tolerance=-0.01)


def test_missing_tolerance_refuses_to_guess(monkeypatch):
    """
    With no calibrated default, the gate must fail rather than invent a number.
    A gate that silently defaults to something plausible is worse than no gate,
    because it looks like it is protecting you.
    """
    import promote

    monkeypatch.setattr(promote, 'DEFAULT_TOLERANCE', None)
    with pytest.raises(ValueError, match='refuses to guess'):
        promote.decide(0.50, 0.50, 0.50, tolerance=None)


def test_verdict_stringifies_for_logs():
    v = decide(0.60, 0.50, 0.50, tolerance=TOL)
    assert str(v).startswith('PROMOTE:')
    assert str(Verdict(promote=False, reason='nope', candidate_f1=0.1)).startswith('REJECT:')


class TestSummariseSpread:
    def test_computes_max_pairwise_gap(self):
        s = summarise_spread([0.50, 0.52, 0.51])
        assert s['n_runs'] == 3
        assert s['min'] == pytest.approx(0.50)
        assert s['max'] == pytest.approx(0.52)
        assert s['max_pairwise_gap'] == pytest.approx(0.02)
        assert s['mean'] == pytest.approx(0.51)

    def test_identical_runs_have_zero_spread(self):
        s = summarise_spread([0.5, 0.5])
        assert s['max_pairwise_gap'] == 0.0
        assert s['stdev'] == 0.0

    def test_single_run_is_not_a_spread(self):
        with pytest.raises(ValueError, match='at least 2'):
            summarise_spread([0.5])
