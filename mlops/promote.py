"""
The promotion gate: decide whether a candidate may replace the champion, and
refuse loudly when it may not.

decide() is pure - measurements in, a verdict out - so it is directly testable
with no models, dataset or mlflow. Everything stateful lives in cli.cmd_promote.

WHAT THE GATE CHECKS, AND WHY IT IS NOT MACRO F1
------------------------------------------------
The first version gated on macro F1 with a tolerance of 0.008. Measuring 10
training seeds on the frozen holdout showed that was unworkable: macro F1 has
sigma = 0.00655 and a range of 0.0237 across runs of the IDENTICAL
configuration, i.e. 6.8% of the metric. A tolerance honestly calibrated to that
noise would permit a ~7% real regression - a formality, not a gate.

The cause is structural, not bad luck. Of the 66 tags, support on the 929-map
holdout ranges from 2 to 345. The 20 rarest carry just 2.8% of all label
instances, and a tag with support 2 swings its own F1 by ~0.3-0.5 when a single
map flips. Macro F1 here is substantially a measurement of coin flips.

Micro F1 pools predictions instead of averaging per-tag scores, so it is far
steadier: sigma = 0.00179, range 0.0055, about 1% of the metric. That is what
the gate is built on.

Micro's known weakness is that it is dominated by high-support tags, so a model
could abandon rare tags without moving it. That hole is closed directly rather
than by also gating on a noisy average: the candidate may not go mute on more
tags than the champion does (see NEVER_PREDICTED_ALLOWANCE).

Macro F1 restricted to tags with support >= 10 was evaluated as a third check
and DROPPED, because it changed zero verdicts across the full validation set -
10 honest reruns plus the 1-epoch regression. It is still logged as a metric;
it is simply not a rule, because a rule that never fires is only a rule to
explain. See VERIFIED.md.
"""
from dataclasses import dataclass

# Measured across 10 training seeds of the identical configuration on the frozen
# 929-map holdout. See VERIFIED.md for the raw per-seed numbers.
MICRO_F1_SIGMA = 0.001793

# Tolerance = TOLERANCE_K * sigma.
#
# k is 4, not the 2 or 3 a "95%/99.7%" reflex would suggest, and the reason is
# specific: the tolerance is applied to the distance from a FIXED champion, and
# that champion is itself one draw from the same noisy distribution. The v1
# champion happens to sit 1.89 sigma ABOVE the seed mean. For a candidate landing
# at a perfectly ordinary mean - 2 sigma to still pass, the tolerance must cover
#
#     (champion - mean) + 2 sigma  =  1.89 sigma + 2 sigma  ~=  3.9 sigma
#
# so k = 4. Verified empirically: k = 4 accepts all 10 honest reruns and rejects
# the 1-epoch model. k = 3 rejects honest reruns.
TOLERANCE_K = 4
DEFAULT_TOLERANCE = TOLERANCE_K * MICRO_F1_SIGMA      # 0.007172

# The ratchet floor, anchored to a FIXED reference rather than the best score
# ever recorded.
#
# Anchoring to best-ever was wrong for a reason worth keeping: best-ever is the
# maximum of many noisy draws, so it is biased upward - here by ~1.9 sigma - and
# it only ever moves up. Every future candidate would be measured against the
# luckiest run that ever happened, and the gate would tighten on its own over
# time until it rejected ordinary reruns. Measured: a 0.008 macro tolerance
# rejected 3 of 10 honest reruns against the real champion, but 7 of 10 once the
# best-ever floor was included.
#
# A fixed reference cannot drift. This is the v1 champion's micro F1 - the
# ensemble the deployed C# app shipped with - so the rule reads: never ship a
# model meaningfully worse than what users already have.
REFERENCE_MICRO_F1 = 0.5569668976135489

# Tags with support >= SUPPORT_FLOOR that the candidate never predicts at all.
# Measured across the 10 seeds this count is 1 or 2 (sigma 0.42) against the
# champion's 1, so +1 admits every honest rerun while still catching a model
# that goes mute on a tag with real support.
#
# Deliberately NOT applied to the full 66-tag count: that is far noisier
# (11..16, sigma 1.34) and a strict "must not increase" rule on it rejects 5 of
# 10 honest reruns - including seed 7, the best model in the set.
NEVER_PREDICTED_ALLOWANCE = 1


@dataclass
class Scores:
    """The measurements the gate needs from one model on the fixed holdout."""
    micro_f1: float
    never_predicted_included: int = None

    def __str__(self):
        n = ('?' if self.never_predicted_included is None
             else self.never_predicted_included)
        return 'micro_f1=%.6f never_predicted(support>=floor)=%s' % (self.micro_f1, n)


@dataclass
class Verdict:
    promote: bool
    reason: str
    candidate: Scores = None
    champion: Scores = None
    tolerance: float = None
    checks: list = None          # [(name, passed, detail)] - every rule, always

    def __str__(self):
        head = 'PROMOTE' if self.promote else 'REJECT'
        return '%s: %s' % (head, self.reason)

    def report(self):
        """Every check with its outcome, so a pass is as auditable as a failure."""
        lines = [str(self)]
        for name, passed, detail in (self.checks or []):
            lines.append('  [%s] %-22s %s' % ('PASS' if passed else 'FAIL', name, detail))
        return '\n'.join(lines)


def decide(candidate, champion=None, reference_micro_f1=REFERENCE_MICRO_F1,
           tolerance=None, never_allowance=NEVER_PREDICTED_ALLOWANCE):
    """
    Should this candidate become the champion?

    candidate             Scores for the candidate
    champion              Scores for the current champion, or None if there is none
    reference_micro_f1    fixed floor that does not move as champions change;
                          None disables the check
    tolerance             how far below a baseline micro F1 may fall (>= 0)
    never_allowance       extra never-predicted tags (support >= floor) tolerated

    Promotes iff every applicable check passes:
      1. micro F1 >= champion  - tolerance      (skipped when there is no champion)
      2. micro F1 >= reference - tolerance      (fixed anchor, never drifts up)
      3. never-predicted among high-support tags <= champion's + allowance
    """
    if tolerance is None:
        tolerance = DEFAULT_TOLERANCE
    if tolerance < 0:
        raise ValueError("Tolerance must be >= 0, got %r" % tolerance)

    checks = []

    if champion is not None:
        floor = champion.micro_f1 - tolerance
        ok = candidate.micro_f1 >= floor
        checks.append(('micro_f1 vs champion', ok,
                       '%.6f vs floor %.6f (champion %.6f - tol %.6f)'
                       % (candidate.micro_f1, floor, champion.micro_f1, tolerance)))
    else:
        checks.append(('micro_f1 vs champion', True, 'no champion registered - skipped'))

    if reference_micro_f1 is not None:
        floor = reference_micro_f1 - tolerance
        ok = candidate.micro_f1 >= floor
        checks.append(('micro_f1 vs reference', ok,
                       '%.6f vs floor %.6f (reference %.6f - tol %.6f)'
                       % (candidate.micro_f1, floor, reference_micro_f1, tolerance)))
    else:
        checks.append(('micro_f1 vs reference', True, 'no reference set - skipped'))

    if (champion is not None
            and candidate.never_predicted_included is not None
            and champion.never_predicted_included is not None):
        limit = champion.never_predicted_included + never_allowance
        ok = candidate.never_predicted_included <= limit
        checks.append(('never-predicted tags', ok,
                       '%d vs limit %d (champion %d + allowance %d)'
                       % (candidate.never_predicted_included, limit,
                          champion.never_predicted_included, never_allowance)))
    else:
        checks.append(('never-predicted tags', True, 'not measurable - skipped'))

    failed = [(name, detail) for name, ok, detail in checks if not ok]

    if failed:
        return Verdict(
            promote=False,
            reason='; '.join('%s: %s' % (n, d) for n, d in failed),
            candidate=candidate, champion=champion, tolerance=tolerance, checks=checks)

    if champion is None:
        reason = ('no champion registered, promoting candidate (%s) as the first one'
                  % candidate)
    else:
        delta = candidate.micro_f1 - champion.micro_f1
        reason = ('micro F1 %.6f is %s the champion %.6f (delta %+.6f, tolerance %.6f)'
                  % (candidate.micro_f1,
                     'better than' if delta >= 0 else 'within tolerance of',
                     champion.micro_f1, delta, tolerance))
    return Verdict(promote=True, reason=reason, candidate=candidate,
                   champion=champion, tolerance=tolerance, checks=checks)


def summarise_spread(scores):
    """
    Turn repeated same-config scores into the numbers that justify a tolerance.

    Reports BOTH stdev and max pairwise gap, but the tolerance is derived from
    stdev. The max gap is an order statistic: its expectation grows roughly as
    sigma * sqrt(2 ln n), so it widens every time another seed is added and never
    settles. stdev converges. The first calibration used the gap and had to be
    redone when a later run fell outside it.
    """
    import statistics

    scores = [float(s) for s in scores]
    if len(scores) < 2:
        raise ValueError("Need at least 2 scores to describe a spread")

    stdev = statistics.stdev(scores)
    return {
        'n_runs': len(scores),
        'scores': scores,
        'mean': statistics.fmean(scores),
        'min': min(scores),
        'max': max(scores),
        'stdev': stdev,
        'max_pairwise_gap': max(scores) - min(scores),
        # For n normal samples the expected range is ~3.08 sigma at n=10; a ratio
        # near that says the spread is ordinary noise rather than one odd run.
        'gap_over_stdev': (max(scores) - min(scores)) / stdev if stdev else 0.0,
    }
