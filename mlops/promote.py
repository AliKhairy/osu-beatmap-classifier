"""
The promotion gate: decide whether a candidate is allowed to replace the
champion, and refuse loudly when it is not.

The decision function here is deliberately pure - no mlflow, no filesystem, no
keras. It takes three floats and returns a verdict, which is what makes it
directly unit-testable in tests/test_gate.py. Everything stateful lives in
cli.py's cmd_promote.

WHY TWO CONDITIONS
------------------
The obvious gate is "candidate must not be worse than the champion by more than
a tolerance". That alone has a slow failure mode: every promotion resets the
baseline, so a run of candidates each 0.004 worse than the one before passes
every check individually while the model gets steadily worse. Ten such
promotions lose 0.04 macro F1 with a green gate the entire way.

So the gate also carries a floor at the best score ever recorded. The incumbent
check catches a single bad candidate; the best-ever floor is what stops the
slow drift. A model may only go meaningfully below the historical best by being
promoted deliberately, not by accumulating tolerances.

WHY THE TOLERANCE IS NOT A ROUND NUMBER
---------------------------------------
Training the identical configuration twice gives different macro F1, because
weight init and batch shuffling differ. A tolerance below that noise floor
rejects honest reruns; one far above it waves real regressions through. So the
tolerance is measured, not chosen: see the three-seed calibration in VERIFIED.md
and DEFAULT_TOLERANCE below.
"""
from dataclasses import dataclass

# MEASURED, not chosen. This is the largest macro F1 gap observed between two
# runs of the identical configuration (5 models, 100 epochs, same architecture,
# same frozen holdout), varying only the training seed:
#
#   shipped champion  0.355539     <- unseeded, same config
#   train_seed 1      0.347545
#   train_seed 2      0.355068
#   train_seed 3      0.350412
#   stdev 0.003840    max pairwise gap 0.007994
#
# Rounded up to 0.008. The derivation matters: a plausible-looking 0.005 was
# tried first and would have REJECTED two of those three honest reruns, because
# it sits below the noise floor. A gate that fails reproductions of its own
# champion is worse than no gate - it teaches you to ignore it.
#
# Four runs is a small sample, so treat 0.008 as a lower bound on the true
# spread rather than a confidence interval. The best-ever floor in decide() is
# what compensates for a tolerance that errs generous: a slightly loose
# tolerance can admit one mediocre candidate, but it cannot accumulate.
#
# Update this ONLY by re-running the calibration (see VERIFIED.md), never by
# picking a number that lets a particular candidate through.
DEFAULT_TOLERANCE = 0.008


@dataclass
class Verdict:
    promote: bool
    reason: str
    candidate_f1: float
    champion_f1: float = None
    best_ever_f1: float = None
    tolerance: float = None

    def __str__(self):
        head = 'PROMOTE' if self.promote else 'REJECT'
        return '%s: %s' % (head, self.reason)


def decide(candidate_f1, champion_f1=None, best_ever_f1=None, tolerance=None):
    """
    Should this candidate become the champion?

    candidate_f1  macro F1 of the candidate on the fixed holdout
    champion_f1   macro F1 of the current champion, or None if there is none
    best_ever_f1  best macro F1 ever recorded, or None to skip the ratchet guard
    tolerance     how much worse than a baseline is acceptable (>= 0)

    Promotes iff BOTH:
        candidate >= champion  - tolerance
        candidate >= best_ever - tolerance
    With no champion, promotes unconditionally: something has to go first.
    """
    if tolerance is None:
        tolerance = DEFAULT_TOLERANCE
    if tolerance is None:
        raise ValueError(
            "No tolerance supplied and DEFAULT_TOLERANCE is unset. The gate "
            "refuses to guess: run the seed calibration and set it.")
    if tolerance < 0:
        raise ValueError("Tolerance must be >= 0, got %r" % tolerance)

    if champion_f1 is None:
        return Verdict(
            promote=True,
            reason=('no champion registered, promoting candidate '
                    '(macro F1 %.4f) as the first one' % candidate_f1),
            candidate_f1=candidate_f1, champion_f1=None,
            best_ever_f1=best_ever_f1, tolerance=tolerance)

    champion_floor = champion_f1 - tolerance
    if candidate_f1 < champion_floor:
        return Verdict(
            promote=False,
            reason=('macro F1 %.4f is worse than the champion %.4f by more than '
                    'the tolerance %.4f (floor %.4f)'
                    % (candidate_f1, champion_f1, tolerance, champion_floor)),
            candidate_f1=candidate_f1, champion_f1=champion_f1,
            best_ever_f1=best_ever_f1, tolerance=tolerance)

    if best_ever_f1 is not None:
        best_floor = best_ever_f1 - tolerance
        if candidate_f1 < best_floor:
            return Verdict(
                promote=False,
                reason=('macro F1 %.4f clears the champion %.4f but falls below '
                        'the best ever recorded %.4f by more than the tolerance '
                        '%.4f (floor %.4f). Refusing to ratchet quality down '
                        'through a chain of within-tolerance promotions.'
                        % (candidate_f1, champion_f1, best_ever_f1, tolerance,
                           best_floor)),
                candidate_f1=candidate_f1, champion_f1=champion_f1,
                best_ever_f1=best_ever_f1, tolerance=tolerance)

    delta = candidate_f1 - champion_f1
    direction = 'better than' if delta >= 0 else 'within tolerance of'
    return Verdict(
        promote=True,
        reason=('macro F1 %.4f is %s the champion %.4f (delta %+.4f, tolerance %.4f)'
                % (candidate_f1, direction, champion_f1, delta, tolerance)),
        candidate_f1=candidate_f1, champion_f1=champion_f1,
        best_ever_f1=best_ever_f1, tolerance=tolerance)


def summarise_spread(scores):
    """
    Turn repeated same-config macro F1 scores into the numbers that justify a
    tolerance. Used by the seed calibration; reported in VERIFIED.md.

    max_pairwise_gap is the headline: the worst disagreement between two honest
    reruns of the same configuration, i.e. the smallest tolerance that does not
    reject a model for noise alone.
    """
    import statistics

    scores = [float(s) for s in scores]
    if len(scores) < 2:
        raise ValueError("Need at least 2 scores to describe a spread")

    return {
        'n_runs': len(scores),
        'scores': scores,
        'mean': statistics.fmean(scores),
        'min': min(scores),
        'max': max(scores),
        'stdev': statistics.stdev(scores),
        'max_pairwise_gap': max(scores) - min(scores),
    }
