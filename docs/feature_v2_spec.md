# v2 feature vector: port spec for `FeatureExtractor.cs`

This is what the OsuScoutNew app has to compute before it can run a v2 model.
`features_v2.py` is the normative source and this document explains it. Where
the two disagree, the code wins, and the goldens decide.

**Status: ported.** `OsuScoutNew/FeatureExtractorV2.cs` (branch `feat/v2-features`
in the app repo) implements everything below. Over all 4961 readable maps in
`downloads/` it matches Python to a worst absolute difference of 9e-13, and run
through the app's own classifier with an exported v2 ensemble it gives the same
tags as Python on every `songs/` map (probabilities within 2e-7). VERIFIED.md
section 16 has the runs.

**Why port it.** v2 beats the shipped v1 vector on the frozen 929-map holdout at
the same 0.27 threshold, on every measure the gate and the per-tag report use.
VERIFIED.md §15 has the numbers:

| | v1 retrains (10 seeds) | v2 (3 seeds) |
|---|---|---|
| micro F1 | 0.557 | **0.582** |
| micro precision | 0.499 | **0.522** |
| false positives | 2847 | **2716** |

## What changes in the app

1. **`FeatureExtractorV2`** computes the 72-value v2 vector described below,
   alongside the unchanged v1 `FeatureExtractor`. The library scan picks one by
   `model_config.json`'s `feature_version`, so old model files keep working.
2. **`OsuClassifier`** validates the length from the config (72 for v2), and
   reads its decision rule from the config too:
   - `threshold` (0.26), applied at display precision: a tag counts when
     `probability >= threshold - 0.5 * 10^-display_decimals`, so a tag shown as
     0.26 is predicted at 0.26. The comparison is made in float32, as numpy
     compares the float32 ensemble output in training.
   - `suppressed_by`: a general tag is dropped when one of the listed specific
     tags is also predicted (e.g. `jumps` beside `large jumps`).
3. **The model files** (5 × `.onnx` + `model_config.json`) come from a v2 training
   run. `model_config.json`'s scaler constants belong to that run, exactly as
   with v1. Export with:

   ```bash
   python cli.py export-onnx --model-dir <promoted dir> --allow-feature-version 2
   ```

   Without that flag the export refuses a v2 model, on purpose.
4. **The tag list** in `model_config.json` is now the 58-label policy's list. It is
   read at runtime, so no code change is needed for it.

The parser also needs `[Difficulty]` and `[TimingPoints]`, which v1 never read.

## Parity checks

- **Golden vector:** `tests/golden_feature_vector_v2.json`, for
  `songs/Polyphia - Playing God (Mir) [Nirvana].osu`, carrying `feature_names`.
- **Three more fixtures:** `python make_goldens.py <dir> --feature-version 2`
  writes `<style>.python.v2.json` for the stream, jump and tech maps the v1
  parity harness already uses.
- **Any map:** `python parity_dump.py --feature-version 2 <map.osu> out.json`.
- **Target:** agreement to 1e-9, the same bar as v1.

## Parsing (mirror `osu_parser.py`)

**Hit objects.** Use the same rules as v1's parser:
- `x, y, time, type` are parsed as **int**.
- For sliders (`type & 2`, with at least 8 fields):
  - the curve type is the first `|`-separated token;
  - control points are `int:int` pairs;
  - `slides` is an int;
  - `length` is a float.
- A line that fails to parse is skipped **whole**. That includes lazer-exported
  sliders whose control points are floats; v1 skips these too, and v2 must match.

**`[Difficulty]`.** Read `key:value` pairs as floats. If `ApproachRate` is
absent, it takes the `OverallDifficulty` value.

**`[TimingPoints]`.** Each line gives `(time = float(field 0), beat_length =
float(field 1), uninherited)`:
- `uninherited` is `field 6 == "1"` when there are at least 7 fields.
- Otherwise (old format) it is `beat_length > 0`.
- Skip `//` comments and blank lines.
- Sort by time **stably**, so equal times keep their file order.

**Missing sections.** If either section is missing, use these defaults:

| Value | Default |
|---|---|
| CS | 4.0 |
| AR | 9.0 |
| OD | 8.0 |
| SliderMultiplier | 1.4 |
| Beat length | 2 × the most common onset interval among intervals strictly between 30 and 1000 ms (rounded to whole ms; ties go to the smallest), or 500 ms if there are none |

## Preprocessing

1. **Drop spinners** (`type & 8`). If fewer than 5 objects remain, return 72 zeros.
2. **Circle radius** `r = 54.4 - 4.48 * CS`.
3. **Beat length at time t.** Use the last uninherited point with
   `beat_length > 0` at or before `t`, or the first such point if `t` precedes
   them all. Clamp its value to **[6, 60000] ms**; the dataset contains 1e-298 ms
   trick lines.
4. **Slider velocity at t.** Walk the sorted points and keep a running value
   `sv`, starting at 1:
   - an uninherited point with `beat_length > 0` resets it to 1;
   - an inherited point with `beat_length < 0` sets it to `clamp(-100 / beat_length, 0.1, 10)`;
   - skip every other point.

   The value at `t` is the running value after the last point at or before `t`,
   or after the first point if `t` precedes them all. If no point qualifies, it
   is 1.
5. **Per object:**
   - `beat` = beat length at its time;
   - `px_per_ms = SliderMultiplier * 100 * sv / beat`.
6. **Sliders.** An object counts as a slider only if `type & 2`, it has a curve
   type, and `length > 0`. For each one:
   - `span = length / px_per_ms`;
   - `end_time = time + span * slides`;
   - if `slides` is **odd**, the end position is the point `length` along its
     path; if **even**, it is the head.
   - **Path point** (`slider_path_end`). For `'P'` with exactly 2 control points
     (3 points in all) and a non-collinear set (|cross| ≥ 1e-6):
     - take the circumcircle;
     - start at the head's angle `atan2(dy, dx)` about the centre;
     - advance `copysign(length / radius, cross)` radians, where
       `cross = (p1 - p0) × (p2 - p1)`.
   - **Anything else** walks the polyline head → control points:
     - skip zero-length segments;
     - if the polyline is shorter than `length`, extend along the last non-zero
       segment's direction;
     - if every segment has zero length, the result is the head.

   This is an **approximation** for Bézier and Catmull curves. It only has to
   match the Python, not osu!.
7. **Transitions** between consecutive objects *i* → *i+1*:

   | Quantity | Definition |
   |---|---|
   | `ioi` | `time[i+1] - time[i]`, head to head |
   | `active` | `ioi <= 2000` |
   | `move` | `head[i+1] - end[i]` |
   | `move_px` | `|move|` |
   | `move_r` | `move_px / r` |
   | `move_ms` | `max(time[i+1] - end_time[i], 10)` |
   | `velocity` | `move_px / move_ms` |
   | `ratio` | `ioi / beat[i]` |

8. **Minutes.** `active_ms` is the sum of `ioi` over active transitions; if it
   is 0, return zeros. Then `minutes = active_ms / 60000` and `n_active` is the
   count of active transitions.
9. **Snap class of `ratio`.** Start at `other`. Then apply these steps in order,
   each overriding the last:
   - if `ratio >= 0.88`, the class is `1_1`;
   - for each d in 2, 3, 4, 6, 8: if `|ratio - 1/d| <= 0.12 / d`, the class is `1_d`;
   - if `ratio < 0.125 * 0.88`, the class is `1_8`.
10. **Angles.**
    - `turn(a, b) = degrees(arccos(clamp(a·b / (|a||b|), -1, 1)))`, using a
      denominator of 1 when `|a||b| = 0`.
    - `angle = 180 - turn`.
    - `cross(a, b) = a.x*b.y - a.y*b.x`, and `sign` returns -1, 0 or +1.

**Numeric conventions** (numpy's defaults, which C# must reproduce):

| Operation | Convention |
|---|---|
| percentile | linear interpolation between closest ranks (numpy default, R type 7) |
| std | population (divide by n) |
| BPM rounding | `np.round`, half to even; C#'s default `Math.Round` also rounds to even |
| empty set | a fraction, mean or percentile over an empty set is **0**, unless a feature says otherwise |

## The 72 features, in order

The following are used throughout:

| Term | Meaning |
|---|---|
| `frac(mask)` | the share of True values; 0 when empty |
| `ratio(a, b)` | `a/b`; 0 when b is 0 |
| `jump` | `active & 0.40 <= ratio <= 1.10 & move_r >= 2` |
| chains | maximal runs of consecutive transitions with `active & ratio <= 0.30`. A run of k transitions is a chain of k+1 notes, covering notes `s..s+k` |

### Tempo and settings (6)

| # | name | definition |
|---|---|---|
| 0 | `cs` | CS |
| 1 | `ar` | AR |
| 2 | `od` | OD |
| 3 | `log_dominant_bpm` | `ln` of the most common value of `round(60000 / beat)` across objects (ties go to the smallest) |
| 4 | `bpm_count` | number of distinct `round(60000 / beat)` values |
| 5 | `log_bpm_range_ratio` | `ln(max / min)` of those rounded BPMs |

### Density (3)

| # | name | definition |
|---|---|---|
| 6 | `notes_per_sec` | `n / (active_ms / 1000)` |
| 7 | `slider_ratio` | share of objects that are sliders |
| 8 | `log_active_minutes` | `ln(1 + minutes)` |

### Rhythm (8)

| # | name | definition |
|---|---|---|
| 9–15 | `snap_frac_1_1` … `snap_frac_other` | share of active transitions in each snap class, in the order `1_1, 1_2, 1_3, 1_4, 1_6, 1_8, other` |
| 16 | `snap_change_rate` | share of consecutive transition pairs, **both** active, whose snap classes differ |

### Chains (7)

With `notes` as each chain's note count:

| # | name | definition |
|---|---|---|
| 17 | `doubles_per_min` | chains with notes = 2, per minute |
| 18 | `triples_per_min` | chains with notes = 3, per minute |
| 19 | `bursts_per_min` | chains with 4 ≤ notes ≤ 8, per minute |
| 20 | `streams_per_min` | chains with 9 ≤ notes ≤ 60, per minute |
| 21 | `deathstreams_per_min` | chains with notes ≥ 61, per minute |
| 22 | `log_longest_chain` | `ln(1 + max notes)` (0 with no chains) |
| 23 | `stream_note_frac` | sum of notes over chains with ≥ 9 notes, divided by `n` |

### Stream shape (8)

A chain's steps are `move_r` over its transitions.

| # | name | definition |
|---|---|---|
| 24 | `burst_spacing_radii` | mean step over chains of 3–8 notes |

Features 25–30 are computed over chains of 9 or more notes, and are all 0 if
there are none:

| # | name | definition |
|---|---|---|
| 25 | `stream_spacing_radii` | mean step |
| 26 | `stream_spaced_frac` | share of steps ≥ 2 radii |
| 27 | `stream_spacing_cv` | per chain `std/mean` (0 when the mean is 0), averaged weighted by step count |
| 28 | `stream_cut_rate` | steps where `step > 3 × chain median` **and** `step > 1`, summed and divided by the total step count |
| 29 | `stream_turn_mean_deg` | mean `turn` between consecutive steps within a chain, both of non-zero length |
| 30 | `stream_sharp_turn_frac` | share of those turns > 90° |

The last feature in this group covers chains of every length:

| # | name | definition |
|---|---|---|
| 31 | `chain_slider_frac` | share of sliders among notes in any chain (notes = 2 or more) |

### Jumps (20)

A **jump pair** is two consecutive jumps k and k+1; its `angle` is taken between
`move[k]` and `move[k+1]`.

| # | name | definition |
|---|---|---|
| 32 | `jump_frac` | `ratio(#jumps, n_active)` |
| 33 | `jump_dist_p50_radii` | p50 of `move_r` over jumps |
| 34 | `jump_dist_p90_radii` | p90 of `move_r` over jumps |
| 35 | `jump_large_frac` | share of jumps with `move_px > 256` |
| 36 | `jump_cross_screen_frac` | share of jumps whose start (`end[i]`) and target (`head[i+1]`) are within 96 px of opposite edges: x < 96 and x > 416, or y < 96 and y > 288, in either direction |
| 37 | `jump_velocity_p50` | p50 of `velocity` over jumps |
| 38 | `jump_velocity_p90` | p90 of `velocity` over jumps |
| 39 | `jump_angle_sharp_frac` | share of pair angles < 60 |
| 40 | `jump_angle_square_frac` | share of pair angles in [75, 105] |
| 41 | `jump_angle_wide_frac` | share of pair angles > 120 |
| 42 | `jump_angle_linear_frac` | share of pair angles > 160 |
| 43 | `jump_angle_mean_deg` | mean pair angle |
| 44 | `jump_angle_std_deg` | population std of the pair angles |
| 45 | `move_reversal_frac` | over consecutive transitions **both** active with `move_r >= 1`: share with `turn > 150` |
| 46 | `back_forth_run_frac` | runs of ≥ 3 consecutive reversals, as flagged in 45 per transition index, divided by `n_active` |
| 47 | `jump_rotation_consistency` | over pairs of *adjacent* jump pairs (k and k+1) where both turns are in (15, 165): share with the same `sign(cross)`. **0.5** if there are none |
| 48 | `jump_vertical_frac` | share of jumps with `|dx| <= tan(20°) |dy|` |
| 49 | `jump_spacing_change` | mean over jump pairs of `|d2 - d1| / ((d1 + d2) / 2)`, using `move_r` |
| 50 | `jump_square_frac` | over jump triples k, k+1, k+2: both angles in [75, 105], the same `sign(cross)` for both turns, and the largest side ≤ 1.25 × the smallest |
| 51 | `jump_closed_shape_frac` | over jump triples starting at k: `head[k+m]` within 1 radius of `head[k]` for some m in {3, 4, 5} (existing objects only), **and** `head[k+2]` is **not** within 1 radius of `head[k]` |
| 52 | `micro_move_frac` | `ratio(#(active & ratio >= 0.40 & 0.5 <= move_r < 2), n_active)` |

### Sliders (11)

Features 53–59 and 62–63 are computed over sliders, and are all 0 if there are
none. `span_ratio = span / beat`.

| # | name | definition |
|---|---|---|
| 53 | `log_slider_velocity_p50` | `ln(1 + p50(px_per_ms))` |
| 54 | `log_slider_velocity_p90` | `ln(1 + p90(px_per_ms))` |
| 55 | `log_slider_length_p50_radii` | `ln(1 + p50(length / r))` |
| 56 | `log_slider_length_p90_radii` | `ln(1 + p90(length / r))` |
| 57 | `slider_repeat_frac` | share of sliders with slides ≥ 2 |
| 58 | `burst_sliders_per_min` | sliders with slides ≥ 3 and `span_ratio <= 0.30`, per minute |
| 59 | `buzz_sliders_per_min` | sliders with slides ≥ 2 and `span_ratio <= 0.10`, per minute |
| 60 | `jump_to_slider_frac` | `ratio(#(jump & object i+1 is a slider), n_active)` |
| 61 | `jump_from_slider_frac` | `ratio(#(jump & object i is a slider), n_active)` |
| 62 | `slider_speed_change_frac` | over consecutive sliders: share with `|Δpx_per_ms| / max(previous, 1e-9) > 0.05` |
| 63 | `log_slider_anchor_mean` | `ln(1 + mean number of control points)` |

### Overlaps (2)

| # | name | definition |
|---|---|---|
| 64 | `overlap_frac` | objects k ≥ 2 with `|head[k] - head[k-2]| / r < 0.2` **and** `|head[k-1] - head[k-2]| / r >= 1`, divided by `n` (0 when n < 3) |
| 65 | `stack_frac` | `ratio(#(active & |head[i+1] - head[i]| < 3 px), n_active)` |

### Difficulty over time (6)

**Windows.** Each object belongs to window `w = floor((time - time[0]) / 4000)`.
A window counts only if it holds at least 2 objects. Order the counted windows
by w; the window index itself is the time axis for the trend.

**Load per window.**
- `speed = count / 4`
- `aim` = mean `velocity` over active transitions whose start object is in the
  window (0 if there are none).

**Per load series**, with `p95` its 95th percentile:

| Measure | Definition |
|---|---|
| spike | `p95 / median` if the median > 0, else 0 |
| sustain | share of windows ≥ `0.75 × p95` if p95 > 0, else 0 |
| trend | Pearson correlation of (window index, load) if there are ≥ 3 windows and the population std exceeds `1e-9 × mean(|load|)`, else 0. Below that the spread is rounding noise, and its "trend" flips with summation order |

| # | name | definition |
|---|---|---|
| 66 | `log_aim_spike` | `ln(1 + spike)` of the aim loads |
| 67 | `log_speed_spike` | `ln(1 + spike)` of the speed loads |
| 68 | `aim_sustain` | sustain of the aim loads |
| 69 | `speed_sustain` | sustain of the speed loads |
| 70 | `aim_trend` | trend of the aim loads |
| 71 | `speed_trend` | trend of the speed loads |
