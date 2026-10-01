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

# Every number below is measured on the CURRENT label space - the 58 labels
# left after the label policy (mlops/labels.py) dropped 5 non-skill tags and
# merged 3 synonym pairs - and at the CURRENT decision rule: threshold 0.26,
# applied at display precision (labels.predicted). Changing either moves all of
# them, so they were re-measured rather than carried over. Earlier values are
# kept in VERIFIED.md, alongside how they were got.

# Measured across 10 training seeds of the identical configuration on the frozen
# 929-map holdout, each trained on the 58 labels (candidates/labels58-seed-N).
# See VERIFIED.md for the raw per-seed numbers. Was 0.001793 on 66 labels at
# 0.27, and 0.002130 on 58 labels at 0.27.
MICRO_F1_SIGMA = 0.002318

# Tolerance = TOLERANCE_K * sigma.
#
# k is not the 2 or 3 a "95%/99.7%" reflex would suggest, and the reason is
# specific: the tolerance is applied to the distance from a FIXED champion, and
# that champion is itself one draw from a noisy distribution. For a candidate
# landing at a perfectly ordinary mean - 2 sigma to still pass, the tolerance
# must cover
#
#     (champion - mean) + 2 sigma
#
# On 66 labels the champion sat 1.89 sigma above the seed mean, so k was 4. On
# the 58 labels at 0.26 it sits 2.58 sigma above the retrained seeds' mean
# (0.561577 vs 0.555591), for two measured reasons: it was a lucky draw to begin
# with, and a model trained on a merged tag scores ~0.002 lower than the max of
# two separately trained ones (VERIFIED.md 14). 2.58 + 2 = 4.58, so k = 5.
#
# Verified empirically: k = 5 accepts all 10 honest 58-label retrains and still
# rejects the 1-epoch model (0.4141) by a wide margin.
TOLERANCE_K = 5
DEFAULT_TOLERANCE = TOLERANCE_K * MICRO_F1_SIGMA      # 0.011590

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
#
# Scored on the current 58-label space at the current threshold: the shipped
# 66-label ensemble's probabilities projected by labels.project_probabilities
# (dropped tags removed, merged tags taking the max of their members). It was
# 0.5569668976135489 on the original 66 labels at 0.27 and 0.5629498176082027
# on 58 labels at 0.27; it moves whenever the question does, because leaving it
# would let candidates through measured against a different question.
REFERENCE_MICRO_F1 = 0.561577293075531

# Tags with support >= SUPPORT_FLOOR that the candidate never predicts at all.
# Measured across the 10 58-label seeds at 0.26 this count is 0 or 1 (sigma
# 0.32) against the champion's 1, so +1 admits every honest rerun while still
# catching a model that goes mute on a tag with real support. (At 0.27: 0 to 2,
# sigma 0.47; on 66 labels: 1 or 2, sigma 0.42.)
#
# Deliberately NOT applied to the full tag count: that is far noisier
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
