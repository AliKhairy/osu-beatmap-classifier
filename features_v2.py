# features_v2.py
"""
The v2 map feature vector.

v1 (neural_model.extract_meaningful_features) stays exactly as it is, because the
shipped desktop app computes it in FeatureExtractor.cs and the current models
were trained on it. v2 exists because measuring v1 against the tags showed it
cannot see several of the patterns the tags name, and mislabels others:

  * Its angle buckets measure the TURN between movement vectors (0 = carry
    straight on) but are named and thresholded as the mapper's ANGLE at the
    middle note (180 - turn). Maps tagged 'sharp angles' score LOWER on
    sharp_angle_ratio. It is also counted over every note, stream notes and
    stacks included, so in practice it detects streams.
  * Stream timing is a fixed 165 ms, so on maps above ~182 BPM ordinary 1/2
    jumps count as streams. Doubles are not detected at all.
  * Distances are measured from a slider's head rather than where the cursor
    leaves it, spinners are treated as notes at the centre of the screen, and
    circle size and timing points are never read.

v2 is computed once per map (no break-split sections, whose std block was zero
for 39% of maps), turns counts into per-minute rates so map length appears
once rather than inside every count, and builds each feature against a tag
definition - echosu's own wording where it has one.

UNITS AND WORDS
  radii     distances divided by the circle radius, r = 54.4 - 4.48 * CS, so
            'overlapping' and 'spaced' mean the same thing at every CS
  IOI       onset interval: head time to head time. Rhythm is judged on it.
  ratio     IOI / beat length, so 0.25 is 1/4 whatever the BPM
  move      the cursor path from where the previous object ENDS (a slider's
            end) to the next head. Aim is judged on it.
  turn      angle between consecutive moves: 0 = straight on, 180 = reverse
  angle     the mapper's angle at the middle note: 180 - turn
  log_      features whose raw value is heavy-tailed carry a log. A handful of
            gimmick maps (a 2970-radius slider, a 517-anchor slider, a
            4830 BPM section) otherwise set the scaler's spread for everyone
            and squash every normal map into a sliver of the range.

Every constant below is a contract with the C# port, exactly like v1's: change
one side, change the other, and regenerate the goldens.
"""
import math

import numpy as np

# --- Fallbacks when a map's [Difficulty] / [TimingPoints] are unavailable ---
# Only the training sidecar can lack them (for the current dataset it covers
# every map). The app always reads the .osu file itself.
DEFAULT_CIRCLE_SIZE = 4.0
DEFAULT_APPROACH_RATE = 9.0
DEFAULT_OVERALL_DIFFICULTY = 8.0
DEFAULT_SLIDER_MULTIPLIER = 1.4
FALLBACK_BEAT_MS = 500.0            # 120 BPM, only if no onset interval is usable
# Red-line beat lengths are clamped to what osu! (lazer) itself allows. Mappers
# use absurd values as tricks - 1e-298 ms appears in this dataset - and
# unclamped, one such line turns a BPM into 5e301 and overflows the scaler.
MIN_BEAT_MS = 6.0
MAX_BEAT_MS = 60000.0

PLAYFIELD_WIDTH_PX = 512
PLAYFIELD_HEIGHT_PX = 384

MIN_OBJECTS = 5                     # below this, every feature is 0
BREAK_MS = 2000                     # onset gaps longer than this are breaks

# --- Rhythm: snap classes, judged on IOI / beat length ---
SNAP_DIVISORS = (2, 3, 4, 6, 8)
SNAP_TOLERANCE = 0.12               # relative error allowed around 1/d
WHOLE_BEAT_MIN_RATIO = 0.88         # a beat or longer
SNAP_CLASSES = ('1_1', '1_2', '1_3', '1_4', '1_6', '1_8', 'other')

# --- Chains: consecutive onsets at 1/4 or faster ---
CHAIN_MAX_RATIO = 0.30              # admits 1/4, 1/6, 1/8; excludes 1/3
BURST_MIN_NOTES = 4                 # echosu: bursts are 'longer than 3'...
BURST_MAX_NOTES = 8                 # ...'and too short to be a stream'
STREAM_MIN_NOTES = 9
DEATHSTREAM_MIN_NOTES = 61          # echosu: 'more than 60 notes'
SPACED_MIN_RADII = 2.0              # echosu: spaced = 'notes no longer overlap'
CUT_SPACING_FACTOR = 3.0            # a step this many times the chain's median...
CUT_MIN_RADII = 1.0                 # ...and at least this far is a cut
SHARP_TURN_DEG = 90.0

# --- Jumps: moves at 1/2-to-1/1 rhythm that clear the previous circle ---
JUMP_MIN_RATIO = 0.40
JUMP_MAX_RATIO = 1.10
JUMP_MIN_RADII = 2.0
LARGE_JUMP_MIN_PX = 256.0           # echosu: 'more than half the width of the play area'
CROSS_SCREEN_EDGE_PX = 96.0         # echosu: 'on the screen edge opposite each other'
ANGLE_SHARP_MAX_DEG = 60.0
ANGLE_SQUARE_MIN_DEG = 75.0
ANGLE_SQUARE_MAX_DEG = 105.0
ANGLE_WIDE_MIN_DEG = 120.0
ANGLE_LINEAR_MIN_DEG = 160.0        # echosu linear aim: 'notes in a straight line'
REAL_TURN_MIN_DEG = 15.0            # below this a turn has no reliable direction
REVERSAL_MIN_TURN_DEG = 150.0       # echosu 1-2: 'back and forth jumps' - the move reverses...
REVERSAL_MIN_RADII = 1.0            # ...at any rhythm, once the move clears the circle it left
BACK_FORTH_MIN_REVERSALS = 3        # a run this long is a back-and-forth pattern, not one snap
VERTICAL_MAX_DEG = 20.0
RETURN_MAX_RADII = 1.0              # 'lands on' an earlier note (closed shapes)
SQUARE_SIDE_TOLERANCE = 0.25        # echosu square: 'four notes ... a square'
CLOSED_SHAPE_MAX_STEPS = 5          # triangles (3), squares (4), stars (5)
MICRO_MIN_RADII = 0.5
MICRO_MAX_RADII = 2.0
MIN_MOVE_MS = 10.0                  # floor for the time a move takes

# --- Sliders ---
SV_MIN = 0.1
SV_MAX = 10.0
SLIDER_SPEED_CHANGE_MIN = 0.05      # relative change between consecutive sliders
BURST_SLIDER_MIN_SLIDES = 3         # echosu: 'repeating multiple times...'
BURST_SLIDER_MAX_RATIO = 0.30       # ...'in quick succession' (1/4 or faster per span)
BUZZ_SLIDER_MIN_SLIDES = 2
BUZZ_SLIDER_MAX_RATIO = 0.10        # echosu: '1/12 snap divisor or higher' (1/12 = 0.083)

# --- Overlaps ---
OVERLAP_MAX_RADII = 0.2             # a note placed on the note two back...
OVERLAP_AWAY_MIN_RADII = 1.0        # ...after moving off it (so not a stack)
STACK_MAX_PX = 3.0

# --- Difficulty over time ---
WINDOW_MS = 4000
WINDOW_MIN_OBJECTS = 2
SUSTAIN_FRACTION = 0.75
# A load series whose spread is below this fraction of its size is flat: the
# spread is floating-point rounding, and correlating rounding with time gives an
# arbitrary 'trend' that even a different summation order flips. One map in the
# dataset has this (relative spread 6e-17; the next flattest is 4e-5).
FLAT_LOAD_REL_STD = 1e-9

FEATURE_NAMES_V2 = [
    # tempo and settings
    'cs', 'ar', 'od', 'log_dominant_bpm', 'bpm_count', 'log_bpm_range_ratio',
    # density
    'notes_per_sec', 'slider_ratio', 'log_active_minutes',
    # rhythm
    'snap_frac_1_1', 'snap_frac_1_2', 'snap_frac_1_3', 'snap_frac_1_4',
    'snap_frac_1_6', 'snap_frac_1_8', 'snap_frac_other', 'snap_change_rate',
    # chains
    'doubles_per_min', 'triples_per_min', 'bursts_per_min', 'streams_per_min',
    'deathstreams_per_min', 'log_longest_chain', 'stream_note_frac',
    # stream shape
    'burst_spacing_radii', 'stream_spacing_radii', 'stream_spaced_frac',
    'stream_spacing_cv', 'stream_cut_rate', 'stream_turn_mean_deg',
    'stream_sharp_turn_frac', 'chain_slider_frac',
    # jumps
    'jump_frac', 'jump_dist_p50_radii', 'jump_dist_p90_radii', 'jump_large_frac',
    'jump_cross_screen_frac', 'jump_velocity_p50', 'jump_velocity_p90',
    'jump_angle_sharp_frac', 'jump_angle_square_frac', 'jump_angle_wide_frac',
    'jump_angle_linear_frac', 'jump_angle_mean_deg', 'jump_angle_std_deg',
    'move_reversal_frac', 'back_forth_run_frac', 'jump_rotation_consistency', 'jump_vertical_frac',
    'jump_spacing_change', 'jump_square_frac', 'jump_closed_shape_frac',
    'micro_move_frac',
    # sliders
    'log_slider_velocity_p50', 'log_slider_velocity_p90', 'log_slider_length_p50_radii',
    'log_slider_length_p90_radii', 'slider_repeat_frac', 'burst_sliders_per_min',
    'buzz_sliders_per_min', 'jump_to_slider_frac', 'jump_from_slider_frac', 'slider_speed_change_frac',
    'log_slider_anchor_mean',
    # overlaps
    'overlap_frac', 'stack_frac',
    # difficulty over time
    'log_aim_spike', 'log_speed_spike', 'aim_sustain', 'speed_sustain', 'aim_trend',
    'speed_trend',
]
FEATURE_COUNT_V2 = len(FEATURE_NAMES_V2)
assert len(set(FEATURE_NAMES_V2)) == FEATURE_COUNT_V2, "duplicate v2 feature name"

# The groups above, by name, for ablation (tools/feature_probe.py) and for the
# C# port spec. Each group is contiguous in FEATURE_NAMES_V2.
_GROUP_BOUNDS = [
    ('tempo', 'cs'), ('density', 'notes_per_sec'), ('rhythm', 'snap_frac_1_1'),
    ('chains', 'doubles_per_min'), ('stream_shape', 'burst_spacing_radii'),
    ('jumps', 'jump_frac'), ('sliders', 'log_slider_velocity_p50'),
    ('overlaps', 'overlap_frac'), ('time_profile', 'log_aim_spike'),
]
FEATURE_GROUPS_V2 = {}
for (_group, _first), (_, _next) in zip(_GROUP_BOUNDS, _GROUP_BOUNDS[1:] + [(None, None)]):
    _stop = FEATURE_NAMES_V2.index(_next) if _next else FEATURE_COUNT_V2
    FEATURE_GROUPS_V2[_group] = FEATURE_NAMES_V2[FEATURE_NAMES_V2.index(_first):_stop]
assert sum(map(len, FEATURE_GROUPS_V2.values())) == FEATURE_COUNT_V2


def circle_radius(cs):
    """osu! circle radius in osu! pixels for a given CircleSize."""
    return 54.4 - 4.48 * cs


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def _fallback_beat_length(times):
    """
    A beat length for a map with no timing points: twice the most common onset
    interval, since that interval is most often a 1/2. Crude, and only ever
    used for the one training map whose .osu file is missing.
    """
    ioi = np.diff(times)
    ioi = np.round(ioi[(ioi > 30) & (ioi < 1000)])
    if ioi.size == 0:
        return FALLBACK_BEAT_MS
    values, counts = np.unique(ioi, return_counts=True)
    return 2.0 * float(values[np.argmax(counts)])


class _Timing:
    """Beat length and slider velocity in effect at any time."""

    def __init__(self, timing_points, object_times):
        points = sorted(timing_points or [], key=lambda p: p[0])
        red = [(t, min(max(bl, MIN_BEAT_MS), MAX_BEAT_MS))
               for t, bl, unin in points if unin and bl > 0]
        if not red:
            red = [(0.0, _fallback_beat_length(object_times))]
        self._red_times = np.array([t for t, _ in red], dtype=float)
        self._red_beats = np.array([bl for _, bl in red], dtype=float)

        # A red point resets slider velocity to 1; a green point sets it to
        # -100 / value. The state after each point, in time order.
        sv_times, sv_values, sv = [], [], 1.0
        for t, bl, unin in points:
            if unin and bl > 0:     # any positive red line, clamped or not, resets SV
                sv = 1.0
            elif not unin and bl < 0:
                sv = min(max(-100.0 / bl, SV_MIN), SV_MAX)
            else:
                continue
            sv_times.append(t)
            sv_values.append(sv)
        self._sv_times = np.array(sv_times, dtype=float)
        self._sv_values = np.array(sv_values, dtype=float)

    @staticmethod
    def _lookup(times, at, values, default):
        if times.size == 0:
            return np.full(len(at), default)
        idx = np.searchsorted(times, at, side='right') - 1
        # Before the first point, the first point applies.
        return values[np.clip(idx, 0, None)]

    def beat_length(self, at):
        return self._lookup(self._red_times, at, self._red_beats, FALLBACK_BEAT_MS)

    def slider_velocity(self, at):
        return self._lookup(self._sv_times, at, self._sv_values, 1.0)


# --------------------------------------------------------------------------
# Slider geometry
# --------------------------------------------------------------------------

def _polyline_point(points, distance):
    """
    The point `distance` along a polyline, extended straight past its last
    segment if the polyline is shorter.

    This is the documented approximation for every non-arc slider: a Bezier or
    Catmull curve is walked along its control points rather than the true
    curve. It only has to agree with the C# port, not with osu! itself.
    """
    seg = np.diff(points, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    keep = lengths > 0
    seg, lengths, starts = seg[keep], lengths[keep], points[:-1][keep]
    if lengths.size == 0:
        return points[0]
    for start, vec, length in zip(starts, seg, lengths):
        if distance <= length:
            return start + vec * (distance / length)
        distance -= length
    return starts[-1] + seg[-1] + seg[-1] / lengths[-1] * distance


def _arc_point(p0, p1, p2, distance):
    """The point `distance` along the circular arc p0 -> p1 -> p2, or None if collinear."""
    ax, ay = p1 - p0
    bx, by = p2 - p1
    cross = ax * by - ay * bx
    if abs(cross) < 1e-6:
        return None
    # Circumcentre of the three points.
    d = 2 * (p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1]))
    s0, s1, s2 = p0 @ p0, p1 @ p1, p2 @ p2
    centre = np.array([
        (s0 * (p1[1] - p2[1]) + s1 * (p2[1] - p0[1]) + s2 * (p0[1] - p1[1])) / d,
        (s0 * (p2[0] - p1[0]) + s1 * (p0[0] - p2[0]) + s2 * (p1[0] - p0[0])) / d,
    ])
    radius = np.linalg.norm(p0 - centre)
    # A left-turning path (cross > 0) goes counter-clockwise, i.e. with
    # increasing atan2 angle.
    theta = math.atan2(p0[1] - centre[1], p0[0] - centre[0])
    theta += math.copysign(distance / radius, cross)
    return centre + radius * np.array([math.cos(theta), math.sin(theta)])


def slider_path_end(head, curve_type, curve_points, length):
    """Where a slider's path ends: `length` along it (exact arc for 'P' sliders)."""
    points = np.array([head] + [list(p) for p in curve_points], dtype=float)
    if curve_type == 'P' and len(points) == 3:
        end = _arc_point(points[0], points[1], points[2], length)
        if end is not None:
            return end
    return _polyline_point(points, length)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _classify_snaps(ratio):
    """Map IOI/beat ratios onto SNAP_CLASSES indices."""
    out = np.full(len(ratio), SNAP_CLASSES.index('other'))
    out[ratio >= WHOLE_BEAT_MIN_RATIO] = SNAP_CLASSES.index('1_1')
    for d in SNAP_DIVISORS:
        ideal = 1.0 / d
        near = np.abs(ratio - ideal) <= SNAP_TOLERANCE * ideal
        out[near] = SNAP_CLASSES.index('1_%d' % d)
    # Anything faster than 1/8 (1/12, 1/16) joins the fastest class.
    out[ratio < (1.0 / 8) * (1 - SNAP_TOLERANCE)] = SNAP_CLASSES.index('1_8')
    return out


def _runs(mask):
    """(start, stop) pairs of each run of True values."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _turn_deg(a, b):
    """Angle between row vectors a and b, in degrees (0 = same direction)."""
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.where(na * nb > 0, na * nb, 1.0)
    return np.degrees(np.arccos(np.clip((a * b).sum(axis=1) / denom, -1.0, 1.0)))


def _cross(a, b):
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _pct(values, q):
    return float(np.percentile(values, q)) if len(values) else 0.0


def _mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def _frac(mask):
    return float(np.mean(mask)) if len(mask) else 0.0


def _ratio(num, den):
    return float(num) / den if den else 0.0


def _load_profile(loads, order):
    """spike, sustain and trend of a per-window load series."""
    if len(loads) == 0:
        return 0.0, 0.0, 0.0
    p95 = float(np.percentile(loads, 95))
    median = float(np.median(loads))
    spike = p95 / median if median > 0 else 0.0
    sustain = float(np.mean(loads >= SUSTAIN_FRACTION * p95)) if p95 > 0 else 0.0
    trend = 0.0
    if len(loads) >= 3 and np.std(loads) > FLAT_LOAD_REL_STD * np.mean(np.abs(loads)):
        trend = float(np.corrcoef(order, loads)[0, 1])
    return spike, sustain, trend


# --------------------------------------------------------------------------
# The feature vector
# --------------------------------------------------------------------------

def extract_features_v2(hit_objects, difficulty=None, timing_points=None):
    """
    The v2 feature vector for one map.

    hit_objects    OsuFileParser.extract_raw_hit_objects() output (or the same
                   lists from ml_dataset.json)
    difficulty     OsuFileParser.get_difficulty() output, or None for defaults
    timing_points  OsuFileParser.get_timing_points() output, or None

    Returns an array of FEATURE_COUNT_V2 floats, in FEATURE_NAMES_V2 order;
    all zeros for a map too short to measure.
    """
    zeros = np.zeros(FEATURE_COUNT_V2)
    difficulty = difficulty or {}

    # Spinners are not aim or tapping; v1 treated them as notes at (256, 192).
    objs = [o for o in hit_objects if not int(o[3]) & 8]
    n = len(objs)
    if n < MIN_OBJECTS:
        return zeros

    cs = float(difficulty.get('CircleSize', DEFAULT_CIRCLE_SIZE))
    ar = float(difficulty.get('ApproachRate', DEFAULT_APPROACH_RATE))
    od = float(difficulty.get('OverallDifficulty', DEFAULT_OVERALL_DIFFICULTY))
    slider_multiplier = float(difficulty.get('SliderMultiplier', DEFAULT_SLIDER_MULTIPLIER))
    radius = circle_radius(cs)

    t = np.array([o[2] for o in objs], dtype=float)
    head = np.array([(o[0], o[1]) for o in objs], dtype=float)
    timing = _Timing(timing_points, t)
    beat = timing.beat_length(t)
    slider_px_per_ms = slider_multiplier * 100.0 * timing.slider_velocity(t) / beat

    # --- where and when each object ends ---
    is_slider = np.zeros(n, dtype=bool)
    slides = np.ones(n)
    length = np.zeros(n)
    anchors = np.zeros(n)
    span_ms = np.zeros(n)
    end = head.copy()
    end_t = t.copy()
    for i, o in enumerate(objs):
        if not (int(o[3]) & 2) or o[4] is None or float(o[7]) <= 0:
            continue
        is_slider[i] = True
        slides[i] = max(int(o[6]), 1)
        length[i] = float(o[7])
        anchors[i] = len(o[5])
        span_ms[i] = length[i] / slider_px_per_ms[i]
        end_t[i] = t[i] + span_ms[i] * slides[i]
        # An even number of slides ends back at the head.
        if int(slides[i]) % 2 == 1:
            end[i] = slider_path_end(head[i], o[4], o[5], length[i])

    # --- transitions between consecutive objects ---
    ioi = np.diff(t)
    active = ioi <= BREAK_MS
    active_ms = float(ioi[active].sum())
    if active_ms <= 0:
        return zeros
    minutes = active_ms / 60000.0
    n_active = int(active.sum())

    move = head[1:] - end[:-1]
    move_px = np.linalg.norm(move, axis=1)
    move_r = move_px / radius
    move_ms = np.maximum(t[1:] - end_t[:-1], MIN_MOVE_MS)
    velocity = move_px / move_ms
    ratio = ioi / beat[:-1]

    f = {}

    # --- tempo and settings ---
    bpm = np.round(60000.0 / beat)
    values, counts = np.unique(bpm, return_counts=True)
    f['cs'], f['ar'], f['od'] = cs, ar, od
    f['log_dominant_bpm'] = math.log(values[np.argmax(counts)])
    f['bpm_count'] = float(len(values))
    f['log_bpm_range_ratio'] = math.log(bpm.max() / bpm.min())

    # --- density ---
    f['notes_per_sec'] = n / (active_ms / 1000.0)
    f['slider_ratio'] = float(is_slider.mean())
    f['log_active_minutes'] = math.log1p(minutes)

    # --- rhythm ---
    snap = _classify_snaps(ratio)
    snap_active = snap[active]
    for k, name in enumerate(SNAP_CLASSES):
        f['snap_frac_' + name] = _frac(snap_active == k)
    both = active[:-1] & active[1:]
    f['snap_change_rate'] = _frac((snap[:-1] != snap[1:])[both])

    # --- chains: runs of 1/4-or-faster onsets ---
    chain_step = active & (ratio <= CHAIN_MAX_RATIO)
    chains = [(s, e, e - s + 1) for s, e in _runs(chain_step)]    # (start, stop, notes)
    notes = np.array([c[2] for c in chains])
    f['doubles_per_min'] = float(np.sum(notes == 2)) / minutes
    f['triples_per_min'] = float(np.sum(notes == 3)) / minutes
    f['bursts_per_min'] = float(np.sum((notes >= BURST_MIN_NOTES)
                                       & (notes <= BURST_MAX_NOTES))) / minutes
    f['streams_per_min'] = float(np.sum((notes >= STREAM_MIN_NOTES)
                                        & (notes < DEATHSTREAM_MIN_NOTES))) / minutes
    f['deathstreams_per_min'] = float(np.sum(notes >= DEATHSTREAM_MIN_NOTES)) / minutes
    f['log_longest_chain'] = math.log1p(notes.max()) if notes.size else 0.0
    f['stream_note_frac'] = float(notes[notes >= STREAM_MIN_NOTES].sum()) / n

    # --- stream shape ---
    burst_steps = [move_r[s:e] for s, e, k in chains if 3 <= k <= BURST_MAX_NOTES]
    stream_chains = [(s, e) for s, e, k in chains if k >= STREAM_MIN_NOTES]
    f['burst_spacing_radii'] = _mean(np.concatenate(burst_steps)) if burst_steps else 0.0
    if stream_chains:
        steps = np.concatenate([move_r[s:e] for s, e in stream_chains])
        cvs, weights, cuts, turns = [], [], 0, []
        for s, e in stream_chains:
            chain = move_r[s:e]
            m = chain.mean()
            cvs.append(chain.std() / m if m > 0 else 0.0)
            weights.append(len(chain))
            median = np.median(chain)
            cuts += int(np.sum((chain > CUT_SPACING_FACTOR * median) & (chain > CUT_MIN_RADII)))
            vec = move[s:e]
            moving = np.linalg.norm(vec, axis=1) > 0
            pair = moving[:-1] & moving[1:]
            turns.append(_turn_deg(vec[:-1][pair], vec[1:][pair]))
        turns = np.concatenate(turns)
        f['stream_spacing_radii'] = float(steps.mean())
        f['stream_spaced_frac'] = _frac(steps >= SPACED_MIN_RADII)
        f['stream_spacing_cv'] = float(np.average(cvs, weights=weights))
        f['stream_cut_rate'] = cuts / len(steps)
        f['stream_turn_mean_deg'] = _mean(turns)
        f['stream_sharp_turn_frac'] = _frac(turns > SHARP_TURN_DEG)
    else:
        for name in ('stream_spacing_radii', 'stream_spaced_frac', 'stream_spacing_cv',
                     'stream_cut_rate', 'stream_turn_mean_deg', 'stream_sharp_turn_frac'):
            f[name] = 0.0
    chain_notes = np.zeros(n, dtype=bool)
    for s, e, _ in chains:
        chain_notes[s:e + 1] = True
    f['chain_slider_frac'] = _frac(is_slider[chain_notes])

    # --- jumps ---
    jump = active & (ratio >= JUMP_MIN_RATIO) & (ratio <= JUMP_MAX_RATIO) & (move_r >= JUMP_MIN_RADII)
    f['jump_frac'] = _ratio(jump.sum(), n_active)
    f['jump_dist_p50_radii'] = _pct(move_r[jump], 50)
    f['jump_dist_p90_radii'] = _pct(move_r[jump], 90)
    f['jump_large_frac'] = _frac(move_px[jump] > LARGE_JUMP_MIN_PX)
    a, b = end[:-1][jump], head[1:][jump]
    lo, hi_x, hi_y = CROSS_SCREEN_EDGE_PX, PLAYFIELD_WIDTH_PX - CROSS_SCREEN_EDGE_PX, \
        PLAYFIELD_HEIGHT_PX - CROSS_SCREEN_EDGE_PX
    cross_screen = (((a[:, 0] < lo) & (b[:, 0] > hi_x)) | ((b[:, 0] < lo) & (a[:, 0] > hi_x))
                    | ((a[:, 1] < lo) & (b[:, 1] > hi_y)) | ((b[:, 1] < lo) & (a[:, 1] > hi_y)))
    f['jump_cross_screen_frac'] = _frac(cross_screen)
    f['jump_velocity_p50'] = _pct(velocity[jump], 50)
    f['jump_velocity_p90'] = _pct(velocity[jump], 90)

    # Pairs of consecutive jumps: the angle at the note between them.
    pair = np.flatnonzero(jump[:-1] & jump[1:])          # jump k and jump k+1
    turn = _turn_deg(move[pair], move[pair + 1])
    angle = 180.0 - turn
    f['jump_angle_sharp_frac'] = _frac(angle < ANGLE_SHARP_MAX_DEG)
    f['jump_angle_square_frac'] = _frac((angle >= ANGLE_SQUARE_MIN_DEG) & (angle <= ANGLE_SQUARE_MAX_DEG))
    f['jump_angle_wide_frac'] = _frac(angle > ANGLE_WIDE_MIN_DEG)
    f['jump_angle_linear_frac'] = _frac(angle > ANGLE_LINEAR_MIN_DEG)
    f['jump_angle_mean_deg'] = _mean(angle)
    f['jump_angle_std_deg'] = float(np.std(angle)) if len(angle) else 0.0
    # echosu 1-2: 'back and forth jumps'. Measured as the cursor reversing
    # direction, over every move that clears the circle it left - not only
    # moves at jump rhythm, which is where most 1-2 reversals were being missed.
    # A run of reversals is the pattern; one reversal is just a sharp angle.
    moving = active & (move_r >= REVERSAL_MIN_RADII)
    rev_pair = np.flatnonzero(moving[:-1] & moving[1:])
    reversal = np.zeros(len(move), dtype=bool)
    reversal[rev_pair] = _turn_deg(move[rev_pair], move[rev_pair + 1]) > REVERSAL_MIN_TURN_DEG
    f['move_reversal_frac'] = _frac(reversal[rev_pair])
    runs = np.array([e - s for s, e in _runs(reversal)])
    f['back_forth_run_frac'] = _ratio(np.sum(runs >= BACK_FORTH_MIN_REVERSALS), n_active)

    # Flow keeps turning the same way; snap zig-zags. Only real turns count,
    # and with none to compare the value is neutral rather than 'zig-zag'.
    real = (turn > REAL_TURN_MIN_DEG) & (turn < 180.0 - REAL_TURN_MIN_DEG)
    side = np.sign(_cross(move[pair], move[pair + 1]))
    consecutive = (pair[1:] == pair[:-1] + 1) & real[1:] & real[:-1]
    f['jump_rotation_consistency'] = (_frac((side[1:] == side[:-1])[consecutive])
                                      if consecutive.any() else 0.5)

    jv = move[jump]
    f['jump_vertical_frac'] = _frac(np.abs(jv[:, 0]) <= math.tan(math.radians(VERTICAL_MAX_DEG))
                                    * np.abs(jv[:, 1]))
    d1, d2 = move_r[pair], move_r[pair + 1]
    f['jump_spacing_change'] = _mean(np.abs(d2 - d1) / ((d1 + d2) / 2))

    # Three jumps in a row: a square has two right angles turning the same way
    # and near-equal sides; a closed shape returns to its first note.
    triple = np.flatnonzero(jump[:-2] & jump[1:-1] & jump[2:])
    if triple.size:
        ang1 = 180.0 - _turn_deg(move[triple], move[triple + 1])
        ang2 = 180.0 - _turn_deg(move[triple + 1], move[triple + 2])
        square_angles = ((ang1 >= ANGLE_SQUARE_MIN_DEG) & (ang1 <= ANGLE_SQUARE_MAX_DEG)
                         & (ang2 >= ANGLE_SQUARE_MIN_DEG) & (ang2 <= ANGLE_SQUARE_MAX_DEG))
        same_way = (np.sign(_cross(move[triple], move[triple + 1]))
                    == np.sign(_cross(move[triple + 1], move[triple + 2])))
        sides = np.stack([move_r[triple], move_r[triple + 1], move_r[triple + 2]], axis=1)
        even = sides.max(axis=1) <= (1 + SQUARE_SIDE_TOLERANCE) * sides.min(axis=1)
        f['jump_square_frac'] = _frac(square_angles & same_way & even)

        returns_early = np.linalg.norm(head[triple + 2] - head[triple], axis=1) / radius < RETURN_MAX_RADII
        closes = np.zeros(len(triple), dtype=bool)
        for steps_back in range(3, CLOSED_SHAPE_MAX_STEPS + 1):
            ok = triple + steps_back < n
            dist = np.full(len(triple), np.inf)
            dist[ok] = np.linalg.norm(head[triple[ok] + steps_back] - head[triple[ok]], axis=1) / radius
            closes |= dist < RETURN_MAX_RADII
        f['jump_closed_shape_frac'] = _frac(closes & ~returns_early)
    else:
        f['jump_square_frac'] = 0.0
        f['jump_closed_shape_frac'] = 0.0

    micro = active & (ratio >= JUMP_MIN_RATIO) & (move_r >= MICRO_MIN_RADII) & (move_r < MICRO_MAX_RADII)
    f['micro_move_frac'] = _ratio(micro.sum(), n_active)

    # --- sliders ---
    if is_slider.any():
        px_per_ms = slider_px_per_ms[is_slider]
        lengths_r = length[is_slider] / radius
        span_ratio = span_ms[is_slider] / beat[is_slider]
        slides_s = slides[is_slider]
        f['log_slider_velocity_p50'] = math.log1p(_pct(px_per_ms, 50))
        f['log_slider_velocity_p90'] = math.log1p(_pct(px_per_ms, 90))
        f['log_slider_length_p50_radii'] = math.log1p(_pct(lengths_r, 50))
        f['log_slider_length_p90_radii'] = math.log1p(_pct(lengths_r, 90))
        f['slider_repeat_frac'] = _frac(slides_s >= 2)
        f['burst_sliders_per_min'] = float(np.sum((slides_s >= BURST_SLIDER_MIN_SLIDES)
                                                  & (span_ratio <= BURST_SLIDER_MAX_RATIO))) / minutes
        f['buzz_sliders_per_min'] = float(np.sum((slides_s >= BUZZ_SLIDER_MIN_SLIDES)
                                                 & (span_ratio <= BUZZ_SLIDER_MAX_RATIO))) / minutes
        # Slider speed changes between consecutive sliders, from SV or BPM alike -
        # either way the player has to re-read how fast the ball moves.
        rel = np.abs(np.diff(px_per_ms)) / np.maximum(px_per_ms[:-1], 1e-9)
        f['slider_speed_change_frac'] = _frac(rel > SLIDER_SPEED_CHANGE_MIN)
        f['log_slider_anchor_mean'] = math.log1p(anchors[is_slider].mean())
    else:
        for name in ('log_slider_velocity_p50', 'log_slider_velocity_p90', 'log_slider_length_p50_radii',
                     'log_slider_length_p90_radii', 'slider_repeat_frac', 'burst_sliders_per_min',
                     'buzz_sliders_per_min', 'slider_speed_change_frac', 'log_slider_anchor_mean'):
            f[name] = 0.0
    # Jumps onto a slider and jumps out of one's end, kept apart: onto separates
    # the 'slider jumps' tag better than either direction pooled.
    f['jump_to_slider_frac'] = _ratio((jump & is_slider[1:]).sum(), n_active)
    f['jump_from_slider_frac'] = _ratio((jump & is_slider[:-1]).sum(), n_active)

    # --- overlaps ---
    if n >= 3:
        back2 = np.linalg.norm(head[2:] - head[:-2], axis=1) / radius
        away = np.linalg.norm(head[1:-1] - head[:-2], axis=1) / radius
        f['overlap_frac'] = float(np.sum((back2 < OVERLAP_MAX_RADII)
                                         & (away >= OVERLAP_AWAY_MIN_RADII))) / n
    else:
        f['overlap_frac'] = 0.0
    stacked = active & (np.linalg.norm(head[1:] - head[:-1], axis=1) < STACK_MAX_PX)
    f['stack_frac'] = _ratio(stacked.sum(), n_active)

    # --- difficulty over time: fixed windows over the map, breaks skipped ---
    window = ((t - t[0]) // WINDOW_MS).astype(int)
    trans_window = window[:-1]
    aim_loads, speed_loads, order = [], [], []
    for w in np.unique(window):
        count = int(np.sum(window == w))
        if count < WINDOW_MIN_OBJECTS:
            continue
        in_w = (trans_window == w) & active
        speed_loads.append(count / (WINDOW_MS / 1000.0))
        aim_loads.append(_mean(velocity[in_w]))
        order.append(w)
    aim_loads, speed_loads = np.array(aim_loads), np.array(speed_loads)
    aim_spike, f['aim_sustain'], f['aim_trend'] = _load_profile(aim_loads, order)
    speed_spike, f['speed_sustain'], f['speed_trend'] = _load_profile(speed_loads, order)
    f['log_aim_spike'] = math.log1p(aim_spike)
    f['log_speed_spike'] = math.log1p(speed_spike)

    assert list(f) == FEATURE_NAMES_V2 or sorted(f) == sorted(FEATURE_NAMES_V2), \
        "v2 features out of step with FEATURE_NAMES_V2"
    return np.array([f[name] for name in FEATURE_NAMES_V2], dtype=float)


def extract_features_v2_from_osu(osu_path):
    """Parse one .osu file and return its v2 vector (None if unreadable)."""
    from osu_parser import OsuFileParser

    parser = OsuFileParser(osu_path)
    if not parser.read_file():
        return None
    return extract_features_v2(parser.extract_raw_hit_objects(),
                               parser.get_difficulty(), parser.get_timing_points())
