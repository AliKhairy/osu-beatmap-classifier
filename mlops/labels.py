"""
The label policy - which echosu tags the model is trained to predict - and the
rule that turns its probabilities into the tags a player sees.

The rule is that a label must name a PLAYING SKILL. echosu's tags are
free-form community votes, so its vocabulary also contains tags that describe
what a map is for, how it feels, or which mod to play it with, plus pairs of
names that people use for the same skill. Training on those spends model
capacity on labels no feature can see, and splits one skill's votes across two
outputs.

So the policy has two parts, applied in this order:

  MERGED_TAGS    two names for one skill -> the established name
  DROPPED_TAGS   not a skill -> removed from the label space

It is applied in exactly two places, which is the point of it living here:
tag_scraper.filter_tags (so future scrapes and rebuilds come out clean) and
split.prepare_dataset (so the dataset already on disk is trained and scored
under the same policy without being rebuilt). A map whose only tags are dropped
keeps its row with no labels, so the positional evaluation split - and with it
the holdout - does not move.

Changing either set changes the label space, which moves every score in the
registry. project_probabilities() is what lets a model trained under an older
policy still be scored against the current one; see scoring.score_on_holdout.
"""
from dataclasses import dataclass

import numpy as np

# Decided 2026-09-30. None of these names a skill a player trains:
#   progressive difficulty  map structure
#   practise                what the map is for
#   comfortable             a feeling
#   dt speed                a mod recommendation
#   fast                    too vague to mean any one skill
DROPPED_TAGS = frozenset({
    'progressive difficulty',
    'practise',
    'comfortable',
    'dt speed',
    'fast',
})

# Two names for one skill, folded into the established one.
MERGED_TAGS = {
    # Decided 2026-09-30. 'alt' is almost entirely inside 'alternating' (32 of
    # its 34 maps carry both). 'snap'/'snap aim' and 'flow'/'flow aim' rarely
    # co-occur, so taggers split their votes between the names; merged, 'flow
    # aim' covers smooth movement in streams as well as in jumps.
    'alt': 'alternating',
    'snap': 'snap aim',
    'flow': 'flow aim',
    # Scrape-time merges that used to be hard-coded in tag_scraper.filter_tags,
    # made to boost two under-represented tags.
    'linear patterns': 'linear aim',
    'star jumps': 'geometric',
    'triangle jumps': 'geometric',
}

assert not DROPPED_TAGS & set(MERGED_TAGS.values()), "a merge target is also dropped"
assert not set(MERGED_TAGS.values()) & set(MERGED_TAGS), "merges must not chain"


# --- From probabilities to the tags a player sees ---------------------------

# The confidence cutoff. Decided 2026-09-30, from the dev-split threshold sweep
# (tools/feature_probe.py --sweep): micro F1 is flat from about 0.26 to 0.34 and
# falls either side, and 0.26 keeps recall up. It was 0.27 before. Every score
# the gate stores is measured at this value, so changing it means re-measuring
# the gate (tools/calibrate_gate.py) - see promote.py.
THRESHOLD = 0.26

# Probabilities are shown to two decimals, and a tag SHOWN as 0.26 must be
# predicted at a 0.26 threshold - otherwise 0.2551 reads "0.26" and is missing.
# So the comparison is made at the display precision: the effective cutoff sits
# half a display step below the threshold.
DISPLAY_DECIMALS = 2


def prediction_cutoff(threshold=THRESHOLD):
    """The raw probability at which a tag counts, for a displayed threshold."""
    return threshold - 0.5 * 10 ** -DISPLAY_DECIMALS


def predicted(probs, threshold=THRESHOLD):
    """Boolean mask: which probabilities are predicted tags. The one rule everything uses."""
    return np.asarray(probs) >= prediction_cutoff(threshold)


# A general tag is hidden when a more specific tag that already says it is
# shown. Presentation only: the network, the gate and every stored metric still
# see the raw predictions, the same way the 'streams' override is kept out of
# the gate. The app matches searches by substring, so hiding 'jumps' next to
# 'large jumps' does not stop a 'jumps' search finding the map.
#
# Decided 2026-09-30, measured on v2's holdout predictions:
#   jumps         predicted with large jumps 57% of the time (community: 29%),
#                 with short jumps 86% (community: 53%). Hiding it moves micro
#                 precision 0.509 -> 0.510: a clean-up, not a trade.
#   high spacing  65% of its predictions come with large jumps, and alone it is
#                 right 33% of the time. The community does use it on its own
#                 (242 of 445 maps), so it is hidden next to large / cross
#                 screen jumps rather than dropped from the label set.
SUPPRESSED_BY = {
    'jumps': ('large jumps', 'short jumps', 'cross screen jumps'),
    'high spacing': ('large jumps', 'cross screen jumps'),
}


def suppress_redundant(tags):
    """Drop the tags SUPPRESSED_BY says a more specific shown tag already covers."""
    shown = set(tags)
    return [t for t in tags if not shown & set(SUPPRESSED_BY.get(t, ()))]


def canonical(tag):
    """The label a raw tag trains as, or None if the policy drops it."""
    tag = MERGED_TAGS.get(tag, tag)
    return None if tag in DROPPED_TAGS else tag


def apply_label_policy(tags):
    """Raw tags for one map -> its labels: merged, dropped, de-duplicated, sorted."""
    return sorted({c for c in map(canonical, tags) if c is not None})


@dataclass
class Projection:
    """How a model's output columns map onto the current label space."""
    groups: list        # per target label, the model columns that feed it
    dropped: list       # model labels the policy removes
    merged: dict        # target label -> the model labels folded into it

    @property
    def identity(self):
        return not self.dropped and not self.merged

    def apply(self, probs):
        """A merged label's probability is the max of its members'."""
        probs = np.asarray(probs)
        return np.stack([probs[:, cols].max(axis=1) for cols in self.groups], axis=1)

    def __str__(self):
        if self.identity:
            return 'identity'
        merged = ', '.join('%s <- %s' % (t, '+'.join(m)) for t, m in self.merged.items())
        return 'dropped [%s]; merged [%s]' % (', '.join(self.dropped), merged)


def project_probabilities(model_classes, target_classes):
    """
    Plan the projection of a model's outputs onto target_classes.

    Only differences the label policy explains are reconciled: labels it drops
    are discarded, and labels it merges are combined. Anything else - a target
    label the model cannot produce, or a model label the policy does not know -
    raises ValueError, because scoring across an unexplained label mismatch
    compares different questions and looks perfectly fine numerically.

    Max, rather than a sum or mean, because the merged tag is predicted exactly
    when the old model would have predicted either member at the same threshold.
    """
    model_classes = list(model_classes)
    target = list(target_classes)
    columns = {c: [] for c in target}
    dropped, unknown = [], []

    for i, tag in enumerate(model_classes):
        label = canonical(tag)
        if label is None:
            dropped.append(tag)
        elif label in columns:
            columns[label].append(i)
        else:
            unknown.append(tag)

    missing = [c for c in target if not columns[c]]
    if missing or unknown:
        raise ValueError(
            "label spaces differ in a way the label policy does not explain: "
            "the model cannot produce %s, and the policy does not know %s"
            % (missing or 'nothing missing', unknown or 'no extra labels'))

    merged = {c: [model_classes[i] for i in cols] for c, cols in columns.items()
              if len(cols) > 1 or model_classes[cols[0]] != c}
    return Projection(groups=list(columns.values()), dropped=dropped, merged=merged)
