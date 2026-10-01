# VERIFIED

Every claim below was produced by running the stated command on this machine.
Anything not actually run is in [Not done](#not-done) or [Caveats](#caveats) —
not asserted anywhere else.

**Sections 4 and 5 are superseded.** They record the first promotion gate, which
used macro F1 with a tolerance of 0.008. That calibration was wrong; section 11
shows why and section 12 documents the gate that replaced it. They are kept
because how the number came out wrong is part of the record.

**Section 12's constants are superseded by section 14.** The label set was cut
to 58 skill tags, which moves every score, so the reference, sigma and k were
re-measured on the new label space. Section 12's method is unchanged.

Environment: Windows 11, Python 3.12.0, TensorFlow 2.21.0, Keras 3.15.0,
scikit-learn 1.9.0, numpy 2.5.0. Dataset `ml_dataset.json`, sha256
`a88572be06c657b0ce4a814d7b60d31340af70e45dcb9e87899474b515a27ef6`.

---

## 1. Dependencies survive the protobuf downgrade

mlflow pins `protobuf<7`; this venv was on 7.35.1. That downgrade had to be
proven safe before anything was built on it, because it would have invalidated
the whole approach.

```
$ pip install -r requirements-mlops.txt -r requirements-dev.txt
$ python -c "import google.protobuf, mlflow, prefect, evidently, pandas; ..."
protobuf    6.33.6        <- downgraded from 7.35.1
mlflow      3.16.1
prefect     3.8.6
evidently   0.7.23
pandas      3.0.6
```

```
$ python -c "import tensorflow as tf; import tf2onnx; ..."
tensorflow   2.21.0 -> import OK
tf2onnx      1.17.0 -> import OK
keras        3.15.0
load_model  -> OK, input (None, 90) output (None, 66)
```

A real conversion, not just an import:

```
$ python -c "from export_to_onnx import convert_models; convert_models(model_dir='.', out_dir=<tmp>, num_models=1)"
Successfully saved <tmp>/ensemble_model_1.onnx

$ python -c "import onnx; onnx.checker.check_model(onnx.load(...))"
onnx.checker: VALID
opset: [('ai.onnx', 15), ('ai.onnx.ml', 2)]
input : input_layer ['unk__6', 90]
output: output_0 ['unk__7', 66]
```

**Result: pass.** TensorFlow and tf2onnx both work under protobuf 6.33.6, and
the exported ONNX keeps the required 90-in / 66-out shape at opset 15.

---

## 2. The evaluation split is fixed and recorded

Claim: two runs produce the same split.

```
$ python -m mlops.split                 # run 1, cold (no feature cache)
$ python -m mlops.split                 # run 2, warm cache
$ python -m mlops.split --no-cache      # run 3, full re-extraction from raw JSON
```

All three produced a byte-identical manifest:

```
  "seed": 42,
  "test_size": 0.2,
  "dataset_sha256": "a88572be06c657b0ce4a814d7b60d31340af70e45dcb9e87899474b515a27ef6",
  "n_rows": 4643,
  "n_features": 90,
  "n_labels": 66,
  "n_holdout": 929,
  "holdout_id_sha256": "1da06c95cc6655a63688b0e60ee3061135d9cf80eadfa602be1717e44c0060e4"
```

```
$ sha256sum manifest_run{1,2,3}.json | awk '{print $1}' | sort -u | wc -l
1                                      # 1 = all identical
$ diff manifest_run1.json manifest_run3.json
                                       # no output: cold vs cold, byte-identical
```

Timings: run 1 `2m9.6s`, run 2 `2.6s` (cache hit), run 3 full re-extraction.
Run 3 matters most — it proves determinism comes from the seed and the data, not
from reusing a cached array.

### The shipped models can be scored fairly on this split

Before registering the existing models as champion, I checked that they
correspond to the current dataset. Re-extracting features today and refitting a
`StandardScaler` reproduces the shipped `ensemble_scaler.pkl`:

```
rows rebuilt now     : (4643, 90)
shipped n_samples_seen: 4643.0
mean  max abs diff   : 1.3877787807814457e-17
scale max abs diff   : 4.163336342344337e-17
labels match         : True
```

Differences at 1e-17 are float summation-order noise, not a data difference. The
existing `train_test_split(..., random_state=42)` at `ensemble_evaluator.py:49`
is that same seed, so the 929-map holdout is data the shipped models never
trained on. (One caveat applies — see [Caveats](#caveats).)

---

## 3. MLflow records what a run did

```
$ python cli.py evaluate --holdout --model-dir . --run-name shipped-models-baseline

Holdout: 929 maps, split hash 1da06c95cc6655a6
macro_f1=0.3555 micro_f1=0.5570 weighted_f1=0.5360 (threshold=0.27, n=929, 12/66 tags never predicted)
Logged MLflow run: 83fc1a2adb2741f590a3e3c41e21f9aa
```

Read back out of the tracking store (`sqlite:///mlflow.db`), not just printed:

```
$ python -c "from mlops.tracking import describe_run; ..."
status : FINISHED

PARAMS:
  dataset          ml_dataset.json
  dataset_sha256   a88572be06c657b0ce4a814d7b60d31340af70e45dcb9e87899474b515a27ef6
  model_dir        .
  n_features       90
  n_labels         66
  num_models       5
  split_hash       1da06c95cc6655a63688b0e60ee3061135d9cf80eadfa602be1717e44c0060e4
  split_seed       42
  test_size        0.2
  threshold        0.27

SUMMARY METRICS:
  macro_f1                 0.35553932434078184
  micro_f1                 0.5569668976135489
  weighted_f1              0.5359830442519735
  samples_f1               0.5471191256779048
  labels_never_predicted   12.0

PER-TAG METRICS: 66 logged (one per label)
  sample: {'f1_tag_1-2': 0.2917, 'f1_tag_aim': 0.7805, 'f1_tag_aim_consistency': 0.5362}

ARTIFACTS: ['per_tag_report.csv', 'split_manifest.json']
```

**Result: pass.** Every field the brief asked for is present: seed, threshold,
feature count and dataset hash as params; micro, macro and per-tag F1 as
metrics; the per-tag report (built with pandas) as a CSV artifact.

This baseline — **macro F1 0.3555** on the shipped models — is the number the
gate compares against.

---

## 4. The gate tolerance was measured, not chosen [superseded]

> **SUPERSEDED.** This section records the first gate, which used macro F1
> with a tolerance of 0.008. That calibration was wrong - see section 11 -
> and the gate was rebuilt in section 12. Kept because how the number was
> got wrong is part of the record, not because it describes current
> behaviour.

A tolerance has to sit above the noise floor of training, or it rejects honest
reruns; far above it, and it waves real regressions through. So it was measured
first. The identical configuration (5 models, 100 epochs, same architecture, same
frozen holdout) was trained three times varying **only** `--train-seed`, then
scored by the same code path on the same 929 maps:

```
$ python cli.py train-ensemble --out-dir candidates/seed-N --epochs 100 --train-seed N
  seed 1  macro_f1=0.347545  micro_f1=0.553012  train 87.5s
  seed 2  macro_f1=0.355068  micro_f1=0.553298  train 62.6s
  seed 3  macro_f1=0.350412  micro_f1=0.553891  train 77.7s
  stdev 0.003797   max pairwise gap 0.007523
```

The shipped champion is the same configuration, so it is a fourth sample of the
same distribution:

```
  champion         0.355539
  n_runs           4
  mean             0.352141
  min / max        0.347545 / 0.355539
  stdev            0.003840
  MAX PAIRWISE GAP 0.007994
```

**Tolerance = 0.008**, the max pairwise gap rounded up. Set in
`mlops/promote.py:DEFAULT_TOLERANCE`.

This is the part worth keeping: a plausible-looking **0.005 was tried first and
is wrong**. Checked against the measured runs, it would have rejected two of the
three honest reruns of the champion's own configuration:

```
Would a 0.005 tolerance have rejected an honest rerun?
   seed 1 0.347545 vs champion-0.005=0.350539 -> REJECTED
   seed 2 0.355068 vs champion-0.005=0.350539 -> ok
   seed 3 0.350412 vs champion-0.005=0.350539 -> REJECTED

And with the measured 0.008?
   seed 1 0.347545 vs champion-0.008=0.347539 -> ok
   seed 2 0.355068 vs champion-0.008=0.347539 -> ok
   seed 3 0.350412 vs champion-0.008=0.347539 -> ok
```

Two honest limits on this number:

- **Four runs is a small sample.** 0.008 is a lower bound on the true spread,
  not a confidence interval. Seed 1 clears the 0.008 floor by 0.000006 — that
  margin is luck, not precision.
- **It errs generous**, which is why the gate does not rely on it alone. The
  best-ever floor (below) is what prevents a slightly loose tolerance from
  accumulating across promotions.

Reproduce with the committed tool (existing `candidates/seed-N/` directories are
reused, so adding seeds to an earlier calibration is cheap):

```bash
python -m tools.calibrate_gate --seeds 1 2 3 4 5 6 7 8 9 10
```

It prints the spread of every candidate gate metric and the derived constants,
and writes `gate_calibration.json`. The constants in `mlops/promote.py` are
measurements; this is the supported way to change them.

---

## 5. Registry and promotion gate [superseded]

> **SUPERSEDED.** This section records the first gate, which used macro F1
> with a tolerance of 0.008. That calibration was wrong - see section 11 -
> and the gate was rebuilt in section 12. Kept because how the number was
> got wrong is part of the record, not because it describes current
> behaviour.

### The shipped models are champion v1

Registered in the **real** registry. This writes `mlflow.db` only — it reads,
scores and logs the artifacts in the repo root, and copies nothing.

```
$ python <register_champion.py>
Shipped models on the fixed split: macro_f1=0.3555 micro_f1=0.5570 ...
Successfully registered model 'osu-tagger'.
Created version '1' of model 'osu-tagger'.
run_id : 3bf1befeb33a41be914fa8a94410b72f
version: 1
champion alias now -> (1, '3bf1befeb33a41be914fa8a94410b72f', 0.35553932434078184)
best_ever macro_f1 -> 0.35553932434078184
```

### The gate, exercised against throwaway state

Every scenario below ran with `MLFLOW_TRACKING_URI=sqlite:///.tmp_mlflow.db`
and `--root-dir <temp copy of the root artifacts>`. The real registry and the
shipped models were not written to — proved by checksum in section 6.

```
SETUP: throwaway DB .tmp_mlflow2.db; temp root seeded with 7 files

################ (d) EMPTY REGISTRY -> first model promotes unconditionally ################
No champion registered yet.
PROMOTE: no champion registered, promoting candidate (macro F1 0.3475) as the first one
Registered version 1 and moved the 'champion' alias to it (run ea565c72b56b4d8a...)
EXIT CODE: 0   (expect: 0)

################ (b) BETTER CANDIDATE vs CHAMPION -> promotes ################
PROMOTE: macro F1 0.3551 is better than the champion 0.3475 (delta +0.0075, tolerance 0.0080)
Registered version 2 and moved the 'champion' alias to it (run f8ca07bbc169456a...)
EXIT CODE: 0   (expect: 0)

################ (c) 1-EPOCH CANDIDATE -> REJECTED, non-zero exit ################
REJECT: macro F1 0.1745 is worse than the champion 0.3551 by more than the tolerance 0.0080 (floor 0.3471)
EXIT CODE: 2   (expect: non-zero)
(c) temp root UNCHANGED after rejection (sha256 1265f7efd0d58fb6) - nothing was overwritten

################ final registry state (throwaway DB) ################
champion version  : 2
champion macro_f1 : 0.355068
best_ever macro_f1: 0.355068
```

**Result: pass.** The weak candidate (`--epochs 1`, macro F1 0.1745 vs the
champion's 0.3551) is rejected with a non-zero exit, and nothing is copied.

The gate also re-scores the champion rather than trusting its stored metric —
in a separate run the stored 0.3550680 and the re-scored 0.3551 agreed, which
is what you want to see. If the dataset had moved they would not, and the
comparison would have been against a number describing a different test set.

An earlier version of this verification ran each command twice (once to capture
output, once to capture the exit code), which advanced the registry two versions
per scenario and compared a candidate against its own prior promotion. The exit
codes were valid but the sequence was misleading, so it was rewritten to the
single-pass form shown above. Both runs agreed on every verdict.

---

## 6. The Prefect flow, both outcomes

Both runs used `--root-dir <temp>` and the throwaway tracking DB, because a
successful flow run ends in a promotion and that must not replace the shipped
models.

### Run 1 — candidate passes, ONNX exported

```
$ python cli.py pipeline --epochs 100 --train-seed 7 \
      --candidate-dir candidates/flow-good --root-dir <temp>
...
Exported 5 ONNX models + model_config.json to <temp>
RUN 1 exit code: 0   (expect 0)
    onnx count: 5  (expect 5)
```

All six files the app loads were produced:

```
    101350  ensemble_model_1.onnx
    101494  ensemble_model_2.onnx
    101494  ensemble_model_3.onnx
    101518  ensemble_model_4.onnx
    101531  ensemble_model_5.onnx
      6875  model_config.json
```

And the config matches the models beside it, which is the reason the two steps
were merged:

```
config scaler_mean matches the scaler beside it : True
config scaler_scale matches                      : True
config tags match the binarizer                  : True
lengths: mean=90 scale=90 tags=66
identical to the currently shipped model_config.json: False   <- correct, different training run
```

### Run 2 — gate blocks, nothing is exported

```
$ python cli.py pipeline --epochs 1 --train-seed 8 \
      --candidate-dir candidates/flow-blocked --root-dir <temp>

Champion re-scored on this split: macro_f1=0.3626 ...
REJECT: macro F1 0.1873 is worse than the champion 0.3626 by more than the tolerance 0.0080 (floor 0.3546)
RuntimeError: Promotion gate rejected the candidate (exit 2). Export is skipped:
  the models currently in <temp> stay in place.
Flow run 'imperious-bison' - Finished in state Failed(...)

RUN 2 exit code: 1   (expect NON-ZERO)
RUN 2 onnx files in temp root: 0  (expect 0 - gate blocked export)
```

**Result: pass.** The block is structural, not conventional: `export_task`
takes its input from `promote_task`, and `promote_task` raises on rejection, so
a rejected candidate cannot reach export even if someone forgets to check.

---

## 7. Drift report

```
$ python cli.py drift --maps songs --out drift_report.html

Extracted features from 31 map(s) in songs
Drifted columns: 88 of 90  (share 0.978, per-column threshold from Evidently's default preset)
Most drifted features:
  mean_max_stream_spacing_variance   0.919
  max_max_stream_spacing_variance    0.825
  max_global_rhythm_variance         0.717
  max_mean_time_gap                  0.704
  mean_avg_spacing_instability       0.689
  mean_mean_time_gap                 0.667
  mean_num_objects                   0.650
  mean_global_rhythm_variance        0.645
Report: drift_report.html    (7.8 MB)
Logged MLflow run: 5ad25fedc4f84f19a3856eabd5b18141
```

**Read this number carefully — 88/90 is not evidence the model is broken.**
It compares 31 hand-picked maps in `songs/` (mostly Extra/Expert difficulties,
chosen as prediction test cases) against 4643 training maps. A tiny, deliberately
unrepresentative current sample against a large reference will show widespread
distributional difference by construction. The useful signal is *which* features
move and whether that set is stable across folders — here it is dominated by
stream-spacing variance and rhythm-variance features, which is consistent with
`songs/` being a harder, more stream-heavy selection than the training corpus.

### A bug this found in my own first implementation

The first version reported `Drifted columns: None of 90 (share 0.5)`. The 0.5
was not a measurement — a generic recursive search for a key named
`drift_share` had found `config.drift_share`, the **threshold** Evidently was
configured with, and reported it as the result. The fix targets the
`DriftedColumnsCount` metric by its config type and reads only its `value`
block. Recorded here because a monitoring tool quietly reporting a plausible
wrong number is the exact failure this phase is supposed to prevent.

---

## 8. Tests and lint

```
$ pytest -q
..............................................                           [100%]
46 passed

$ python -m ruff check .
All checks passed!
```

Run as bare `pytest`, which is what CI runs — see section 10 for why that
distinction mattered.

`tests/test_gate.py` (25 tests) covers `promote.decide`, in four groups:

- **The calibrated constants themselves** — that `DEFAULT_TOLERANCE` really is
  `TOLERANCE_K * MICRO_F1_SIGMA`, that `k` is 4 rather than a rounding of "about
  3 sigma", and that the reference is the fixed v1 champion score. If someone
  edits a constant without redoing the calibration, this is where it surfaces.
- **The micro F1 checks** — clearly better, worse within tolerance, worse beyond
  it, the inclusive boundary at exactly `champion - tolerance` and one ULP
  below, no champion, and no champion with a bad model (an empty registry is not
  a licence to ship anything: the fixed reference still applies).
- **The ratchet guard** — a candidate that beats a degraded champion but fails
  the fixed reference must be rejected; the reference can be disabled
  explicitly; and a simulated chain of ten successive as-bad-as-allowed
  promotions cannot drift below the reference. This replaces the earlier
  `best_ever - tolerance` rule, which was removed: best-ever is the maximum of
  many noisy runs, so it is biased upward and only ever rises (section 12).
- **The never-predicted guard and reporting** — going mute on extra supported
  tags rejects, the measured +1 allowance admits ordinary variation, one beyond
  it rejects, the rule is skipped when not measurable, and every check is
  recorded on a pass as well as a failure.

Plus tolerance validation (negative rejected, zero legal but strict) and
`summarise_spread` reporting both stdev and max pairwise gap — both, because the
tolerance is derived from stdev while the gap is what misled the first
calibration, and keeping both visible makes that mistake harder to repeat.

`tests/test_features.py` (21 tests) covers the extractor, including a
**golden-vector regression**: `tests/golden_feature_vector.json` pins the exact
90 floats produced for a committed map (`Polyphia - Playing God (Mir)
[Nirvana].osu`, 918 hit objects, 2 sections), compared at `rtol=atol=1e-9`.

That test paid for itself immediately. Ruff found two dead variables
(`x_spread` / `y_spread`) in the collinearity loop of
`neural_model.extract_meaningful_features` — computed for every 4-object chunk
of every section and never read. They were removed, and the golden test
confirmed all 90 floats were unchanged. That is the difference between
believing a change was safe and verifying it.

**If that test ever fails, the fix is not to regenerate the golden.** It means
the feature math moved, which is only correct if `FeatureExtractor.cs` moved
identically and both sets of goldens were regenerated together.

### The pipeline runs on the committed sample dataset

`ml_dataset.json` is ~528 MB and not in the repo, so `samples/sample_features.csv`
(60 already-extracted feature vectors + tags, all 66 labels covered, drawn only
from training rows — zero overlap with the holdout) exists so a reader can run
the pipeline on a fresh checkout:

```
$ python cli.py train-ensemble --dataset samples/sample_features.csv \
      --out-dir candidates/sample-run --epochs 5 --train-seed 1
[split] Extracting features from samples/sample_features.csv (no cache for 6703bab512d6c9b0)...
Training seed: 1 (evaluation split seed stays 42)
Model 1 saved as candidates/sample-run\ensemble_model_1.keras.
... (5 models)
             micro avg       0.10      1.00      0.19        83
             macro avg       0.10      0.62      0.18        83
Artifacts written to: candidates/sample-run
```

**The machinery runs; the model is meaningless.** 60 maps across 66 labels
produces the numbers above, and they should not be read as a result. The file
contains derived statistics and tag labels only — no beatmap content.

### The exit-code contract still holds

Every new subcommand follows the existing rule: 0 on success, non-zero on
failure, so `a && b` chains and CI can rely on it.

```
$ python cli.py promote --candidate /nonexistent      -> exit 1
$ python cli.py drift --maps /nonexistent             -> exit 1
$ python cli.py evaluate --holdout --dataset /nope.json -> exit 1
$ python cli.py definitely-not-a-verb                 -> exit 2
$ python cli.py export-onnx --model-dir /nonexistent  -> exit 1
```

One pre-existing hole in that contract was closed. `cmd_rebuild` returned 0 no
matter what happened inside `rebuild()` — missing token, absent `downloads/`,
nothing matched — so a scripted `cli.py rebuild && cli.py train-ensemble` would
proceed to train on whatever stale dataset was lying around:

```
$ mv .env .env.bak && python cli.py rebuild
Error: ECHO_API_TOKEN not found. Please create a .env file.
Rebuild did not produce a dataset.
exit 1                                    # was 0 before this change
```

`rebuild()` also now writes through `dataset_builder.save_dataset`, which writes
atomically and refuses to overwrite a good dataset with an empty one. It
previously used a plain `json.dump`, so a crash mid-write destroyed a file that
costs hours of scraping, and a run matching nothing silently truncated it to `[]`.

All 10 subcommands (`build-dataset rebuild train train-ensemble export-onnx
predict evaluate promote drift pipeline`) respond to `--help` without importing
TensorFlow, mlflow, prefect or evidently — the lazy-import discipline the
original CLI established.

---

## 9. The shipped models were never touched

Checksums captured **before** any training ran, and re-checked after the
calibration, the champion registration and every gate scenario:

```
$ sha256sum ensemble_model_{1..5}.keras ensemble_scaler.pkl ensemble_binarizer.pkl beatmap_classifier.pkl
88b69d019ec6bf21a097d3c980bce993fcf5dae8bb2e7d7549295341126da645 *ensemble_model_1.keras
0a255b05693ee1f95c3427ac45fdbb2b9de59f3561f951d432d5ee0dd759878e *ensemble_model_2.keras
4d6ab140748d2a0937ebce279f066fa6cd492cbe1565dd668a4b1b792afdee5f *ensemble_model_3.keras
90191d4559b3effe5dcbf11e96970e92654daf199c8f785dd906b6162910c4cd *ensemble_model_4.keras
0eca84c2e56fd7af3656f8e5d439b52cf672e31dd10bed2053c85145201d3bd2 *ensemble_model_5.keras
93b12285bf282417d4b42f9c56873bad7f50188462a977aa9404e2e86b7896ad *ensemble_scaler.pkl
e03ccd3f3753a5cc88fb619c87a51a8fc9dec6a0c643c26900f97061943d5248 *ensemble_binarizer.pkl
bdc1873e3625be781827206039450667d5f073bc5d75e023b8e3778d38c72864 *beatmap_classifier.pkl

$ diff baseline_artifact_hashes.txt now_hashes.txt
IDENTICAL - all 8 shipped artifacts unchanged since before any training ran
```

Every candidate trained into `candidates/<name>/`, and every promotion test
wrote to a temp directory via `--root-dir`.

**Decision: no candidate was promoted.** The shipped ensemble scores higher on
the gate metric than every one of the ten retrains:

```
champion micro_f1       : 0.556967
best of the 10 retrains : 0.556315  (seed 7)
champion beats all 10   : True
margin over the best    : +0.000652
```

So replacing it would make the deployed model slightly worse, and the gate would
be being used to wave through a regression it was built to stop. The shipped
models stay as they are.

Worth noting the two metrics disagree about which model is best: seed 7 has the
higher macro F1 (0.362552 vs the champion's 0.355539) while the champion has the
higher micro F1. That is the macro-vs-micro tension from section 12 showing up
in a real decision rather than in the abstract — and with macro's sigma at
0.00655, seed 7's macro advantage is well inside the noise, while the champion's
micro advantage is small but measured on the steadier metric.

---

## 10. Verified by CI rather than locally

The Docker image build could not be run on the dev machine — Docker Desktop's
daemon was not running (`failed to connect to the docker API at
npipe:////./pipe/dockerDesktopLinuxEngine`). It is covered by CI on PR #2
instead, where every step passed:

```
success  Build image
success  CLI starts and lists its subcommands
success  Every subcommand is wired up            (all 10, incl. promote/drift/pipeline)
success  Unknown subcommand is rejected
success  Missing prerequisites exit non-zero
success  The promotion gate refuses to run on nothing
success  No secrets or bulk data in the image    (incl. the new mlruns/mlflow.db checks)
```

Both jobs green: `Build image and smoke-test the CLI` 1m52s,
`Lint and unit tests` 1m0s.

### CI caught a bug in the verification itself

The first CI run **failed**, and the cause was my own testing method, not the
code:

```
tests/test_gate.py:12: in <module>
    from mlops.promote import Verdict, decide, summarise_spread
E   ModuleNotFoundError: No module named 'mlops'
```

`python -m pytest` puts the current directory on `sys.path`; the bare `pytest`
console script does not, and pytest's default prepend import mode only adds
`tests/`. Every local run in this branch used the `-m` form, so "37 tests pass"
was only ever true for that one invocation — the suite had never worked under
the command CI runs.

Fixed with `pythonpath = ["."]` in `[tool.pytest.ini_options]`, which fixes both
invocations rather than bending the CI command to match a local habit. Verified
under bare `pytest`, `python -m pytest`, and bare `pytest` from an unrelated
working directory: 37 passed in each. Recorded because the lesson generalises —
a verification that only runs one way has not been verified.

(The counts in this section are the suite as it stood at that point. Rebuilding
the gate in section 12 replaced the gate tests; the suite is 46 tests now, as
section 8 shows.)

---

## 11. Recalibration: 10 seeds, and why 0.008 was wrong

The 4-sample estimate in section 4 was not merely a lower bound, it was
misleading. Training seed 7 (run later by the flow verification) scored macro F1
0.3626, outside the range of all four original samples. Recalibrated over **10
training seeds** on the same frozen 929-map holdout, scored by the same code
path:

```
seed   macro_f1     micro_f1        seed   macro_f1     micro_f1
1      0.347545     0.553012        6      0.345831     0.552480
2      0.355068     0.553298        7      0.362552     0.556315
3      0.350412     0.553891        8      0.352205     0.551966
4      0.345370     0.552721        9      0.338858     0.550834
5      0.349700     0.556060        10     0.355478     0.555224
```

```
MACRO F1  mean 0.350302  stdev 0.006553  min 0.338858  max 0.362552  gap 0.023694  (6.76%)
MICRO F1  mean 0.553580  stdev 0.001793  min 0.550834  max 0.556315  gap 0.005482  (0.99%)
```

**The macro spread is 0.0237 - three times the shipped 0.008.** Checked against
the real runs, a 0.008 tolerance rejects honest reruns of the champion's own
configuration, and how many depends on which baseline it is measured against:

```
vs the shipped champion 0.355539 (floor 0.347539): rejects 3 of 10 -> seeds 4, 6, 9
  plus the best-ever floor  0.362552 (floor 0.354552): rejects 7 of 10 -> seeds 1,3,4,5,6,8,9
```

**3 of 10** is the honest figure for the gate as it would run today against the
real champion. **7 of 10** is what happens once the best-ever floor is included,
and that difference is the evidence against anchoring a floor to best-ever:
best-ever here is seed 7, a single lucky draw sitting 1.87 sigma above the mean,
and anchoring to it more than doubles the honest-rerun rejections. Seed 1
(0.347545) clears the champion floor by 0.000006 - that margin is luck, not
precision.

### Tolerance basis: stdev, not max pairwise gap

The original calibration used the max pairwise gap. That is an order statistic:
its expectation grows with the number of runs (~ `sigma * sqrt(2 ln n)`), so it
widens every time a seed is added and never converges. Standard deviation is a
consistent estimator. Both are reported; `k * sigma` is the basis used.

```
metric                  sigma   2sigma   3sigma   maxgap  gap/sigma
macro_f1              0.00655   0.0131   0.0197   0.0237       3.62
macro_f1_supp5        0.00510   0.0102   0.0153   0.0173       3.40
macro_f1_supp10       0.00561   0.0112   0.0168   0.0179       3.20
macro_f1_supp20       0.00489   0.0098   0.0147   0.0154       3.14
macro_f1_supp30       0.00417   0.0083   0.0125   0.0112       2.68
weighted_f1           0.00329   0.0066   0.0099   0.0112       3.39
micro_f1              0.00179   0.0036   0.0054   0.0055       3.06
```

The expected range of 10 normal samples is ~3.08 sigma. Every observed
gap/sigma sits near that, so these spreads are consistent with ordinary
Gaussian training noise rather than anything structural in a particular seed.

### Why macro F1 is the noisy one

This is a property of the label distribution, not bad luck. Across the 66 tags
support runs 2..345 (median 48), and:

```
top-20 by support : mean f1 0.5780 | total support 3206
bottom-20 by supp : mean f1 0.0979 | total support  131

the 20 rarest tags carry 2.8% of all label instances
```

A tag with support 2 moves its own F1 by ~0.3-0.5 when a single holdout map
flips, and that enters macro at 1/66 weight each. Macro F1 here is substantially
a measurement of coin flips on tags that carry almost no signal.

### Choosing the support floor, and a disclosure

**Disclosure first: this is not a blind pre-registration.** The spreads for
floors 5/10/20/30 were computed and seen before this rationale was written. It
is recorded here so the reasoning can be checked against the outcome rather
than presented as if it preceded it.

**The criterion, which is a property of the holdout, not of the gate:**

A per-tag F1 is only worth averaging if it is not dominated by single-map
flips. On a 929-map holdout, one map changes a tag's recall by `1/support`. For
a tag's F1 to be stable to roughly 10% - the level at which averaging 40-odd of
them produces a meaningful number - that tag needs **support >= 10**.

Applied to this holdout: support ranges 2..345, median 48. A floor of 10 keeps
**49 of 66 tags (74%)**, and drops tags where a single map is more than 11% of
the entire class. The dropped tags are the ones whose F1 is closer to a coin
flip than a measurement.

**The check that this was not chosen to flatter the gate:** support >= 30 gives
the *lowest* spread of the four floors (2.39% vs 3.95% relative). If the floor
had been picked to make the gate look best, it would have been 30, not 10.
Floor 10 is chosen because 1/support crosses ~10% there, and it is accepted
despite being the noisier option.

**What the floor gives up, and how that is covered:** averaging only high-support
tags means a model could stop predicting a low-support tag entirely without
moving the metric. That is exactly the failure macro F1 was there to catch, so
the floor is paired with a hard rule on `labels_never_predicted` rather than
used alone.

---

## 12. The final gate, and its validation

Rebuilt from the 10-seed measurements in section 11. It replaces the macro F1
gate of sections 4 and 5, which is kept there only as a record of the mistake.

### The design

```
metric              micro_f1
sigma (10 seeds)    0.001793
k                   4
tolerance = k*sigma 0.007172
fixed reference     0.556967   (the v1 champion; does not drift)
never-predicted     champion's count among support>=10 tags, +1
```

A candidate is promoted iff **all** of:

1. `micro_f1 >= champion  - tolerance` (skipped when there is no champion)
2. `micro_f1 >= reference - tolerance` (fixed anchor)
3. never-predicted tags with support >= 10 `<= champion's + 1`

### Why micro F1, not macro

Macro F1 has sigma 0.00655 and a range of 0.0237 across runs of the *identical*
configuration - 6.8% of the metric. Any tolerance honestly calibrated to that
noise permits roughly a 7% real regression, which is a formality rather than a
gate. Micro F1 has sigma 0.00179, about 1% of the metric, so a tolerance
calibrated to it still bites.

### Why k = 4, not 2 or 3

Not a rounding of "about 3 sigma". The tolerance is applied to the distance from
a **fixed** champion, and that champion is itself one draw from the same noisy
distribution - the v1 champion sits **1.89 sigma above the seed mean**. For a
candidate landing at an unremarkable `mean - 2 sigma` to still pass:

```
tolerance >= (champion - mean) + 2 sigma = 1.89 sigma + 2 sigma ~= 3.9 sigma  ->  k = 4
```

Confirmed empirically: k = 3 rejects honest reruns, k = 4 does not.

### Why a fixed reference instead of best-ever

Best-ever is the maximum of many noisy draws, so it is biased upward - here by
1.87 sigma - and it only ever rises. Anchoring to it measures every future
candidate against the luckiest run that ever happened, and the gate silently
tightens over time. Section 11 shows the cost directly: a 0.008 macro tolerance
rejected 3 of 10 honest reruns against the real champion, but 7 of 10 once the
best-ever floor was added.

A fixed reference cannot drift. Set to the v1 champion's micro F1, so the rule
reads: never ship a model meaningfully worse than what users already have.

### What was dropped, and why

`macro_f1_supported` (macro F1 over the 49 tags with support >= 10) was
evaluated as a third check and **dropped**: across the full validation set - 10
honest reruns plus the 1-epoch regression - it changed **zero** verdicts. It is
still logged as a metric for diagnosis; it is simply not a rule, because a rule
that never fires is only a rule to explain.

A rule on the *total* never-predicted count was also rejected. That count swings
11..16 across identical reruns (sigma 1.34), and a strict "must not increase"
rejects 5 of 10 honest reruns - including seed 7, the best model in the set.
Restricted to tags with support >= 10 the count is 1..2 (sigma 0.42), which is
what makes rule 3 workable.

### Validation

Every model measured in this branch, through the real `mlops.promote.decide()`,
against the v1 champion:

```
subject                micro_f1    never  verdict   expected
------------------------------------------------------------------------
honest seed 1          0.553012    1      PROMOTE   promote
honest seed 2          0.553298    1      PROMOTE   promote
honest seed 3          0.553891    1      PROMOTE   promote
honest seed 4          0.552721    1      PROMOTE   promote
honest seed 5          0.556060    1      PROMOTE   promote
honest seed 6          0.552480    1      PROMOTE   promote
honest seed 7          0.556315    2      PROMOTE   promote
honest seed 8          0.551966    1      PROMOTE   promote
honest seed 9          0.550834    2      PROMOTE   promote
honest seed 10         0.555224    1      PROMOTE   promote
1-epoch regression     0.414016    2      REJECT    reject

margin of the WORST honest rerun:
  honest seed 9  micro_f1 0.550834  floor 0.549795  clears by +0.001039

how far the 1-epoch model is from passing:
  micro_f1 0.414016  floor 0.549795  short by 0.135779 (18.9 x tolerance)

RESULT: PASS (0 wrong verdicts out of 11)
```

The worst honest rerun clears by 0.58 sigma and the regression fails by 18.9x
the tolerance, so neither outcome is marginal.

### End to end through the real command

Not just the pure function - the actual `cli.py promote`, against a throwaway
tracking DB and a temp `--root-dir`.

Empty registry. Note the fixed reference is checked even with no champion, so
an empty registry is not a licence to ship anything:

```
$ python cli.py promote --candidate candidates/seed-1 --root-dir <temp>

PROMOTE: no champion registered, promoting candidate (micro_f1=0.553012 never_predicted(support>=floor)=1) as the first one
  [PASS] micro_f1 vs champion   no champion registered - skipped
  [PASS] micro_f1 vs reference  0.553012 vs floor 0.549795 (reference 0.556967 - tol 0.007172)
  [PASS] never-predicted tags   not measurable - skipped
Registered version 1 and moved the 'champion' alias to it
EXIT: 0
```

The 1-epoch regression against that champion:

```
$ python cli.py promote --candidate candidates/weak-1epoch --root-dir <temp>

Candidate: micro_f1=0.4140 macro_f1=0.1745 ... never-predicted 2/49 supported, 9/66 overall
Champion re-scored on this split: micro_f1=0.5530 macro_f1=0.3475 ... never-predicted 1/49 supported, 12/66 overall

REJECT: micro_f1 vs champion: 0.414016 vs floor 0.545840; micro_f1 vs reference: 0.414016 vs floor 0.549795
  [FAIL] micro_f1 vs champion   0.414016 vs floor 0.545840 (champion 0.553012 - tol 0.007172)
  [FAIL] micro_f1 vs reference  0.414016 vs floor 0.549795 (reference 0.556967 - tol 0.007172)
  [PASS] never-predicted tags   2 vs limit 2 (champion 1 + allowance 1)
EXIT: 2
temp root UNCHANGED after rejection
```

That third line is the documented limitation made visible rather than hidden:
rule 3 **passes** the 1-epoch model, because an undertrained model fires more
tags, not fewer. Rule 1 is what rejects it. Every check is reported on a pass as
well as a failure, so what the gate did and did not catch is always legible.

### The constants are reproducible from the committed tool

The numbers in `mlops/promote.py` are not transcribed from a scratch script that
no longer exists. `tools/calibrate_gate.py` is committed, and re-running it over
the ten candidate directories reproduces them exactly:

```
$ python -m tools.calibrate_gate --seeds 1 2 3 4 5 6 7 8 9 10

metric                      mean     sigma       min       max       gap  gap/sig
micro_f1                0.553580  0.001793  0.550834  0.556315  0.005482     3.06
macro_f1                0.350302  0.006553  0.338858  0.362552  0.023694     3.62
macro_f1_supported      0.453592  0.005607  0.447350  0.465270  0.017920     3.20
weighted_f1             0.532557  0.003295  0.526996  0.538154  0.011158     3.39

never_predicted_supported across seeds: [1,1,1,1,1,1,2,1,2,1] (champion 1, sigma 0.42)
  smallest allowance admitting every run: +1

=== derived gate constants (micro_f1) ===
  MICRO_F1_SIGMA           = 0.001793     <- matches mlops/promote.py
  TOLERANCE_K              = 4
  DEFAULT_TOLERANCE        = 0.007172     <- matches mlops/promote.py
  NEVER_PREDICTED_ALLOWANCE= 1            <- matches mlops/promote.py
  worst run 0.550834 vs champion floor 0.549795 -> accepted
```

A calibration whose script is lost is just a number someone once believed.

### Honest limits

- **The rare tags are largely unprotected.** Of the 17 tags below the support
  floor, 11 are already never predicted by the champion. Rule 3 guards the 49
  supported tags; it cannot protect tags the model had already abandoned.
- **Rule 3 does not catch undertrained models.** The 1-epoch model has *fewer*
  never-predicted tags than the champion (its `never_predicted_included` is 2,
  within the allowance) because it fires indiscriminately above the threshold.
  It is caught by rule 1, overwhelmingly. Rule 3 guards a different failure: a
  model going selectively mute.
- **Ten seeds is better than four, not definitive.** sigma is a consistent
  estimator so it will not keep widening the way the max gap did, but it is
  still estimated from 10 samples.

---

## 13. The hard constraints were not violated

The brief forbids changing the 90-feature vector, its order, the scaler, or the
0.27 threshold, because the C# app and its parity goldens depend on them. A text
diff cannot show this cleanly - line-ending normalisation rewrote whole files -
so it is verified behaviourally instead.

**The feature vector is bitwise identical to `origin/main`.** Both versions of
the extractor were loaded side by side in one process and run over the same
committed map:

```
$ git show origin/main:neural_model.py > <tmp>/neural_model.py
$ python  # import both, extract the same map with each

origin/main : len 90, 2 sections
this branch : len 90, 2 sections
shapes equal      : True
BITWISE identical : True
max abs difference: 0.000e+00
```

Not "close to", not "within tolerance" - the same floats. The only changes to
`neural_model.py` are additive (`BASE_FEATURE_NAMES` / `FEATURE_NAMES`, which
nothing computes from), the removal of two provably dead variables, and comment
or loop-variable renames.

The other three constraints:

- **Feature order** - implied by the bitwise result above, and independently
  asserted by `AGGREGATED_FEATURE_COUNT == FEATURE_COUNT * 3 + 3` plus the
  golden-vector test.
- **The scaler** - still `StandardScaler` fit on all rows before splitting
  (`mlops/split.py:scale_all`), including the pre-existing leakage, which was
  deliberately preserved. Section 2 shows a refit reproduces the shipped
  `ensemble_scaler.pkl` to 1e-17.
- **The 0.27 threshold** - now a named constant (`ensemble_evaluator.THRESHOLD`)
  instead of a scattered literal, with the same value, pinned by a test.

---

## 14. The label policy, and re-measuring the gate for it

Added 2026-09-30. The label set was cut to skills only (`mlops/labels.py`): five
tags dropped (progressive difficulty, practise, comfortable, dt speed, fast) and
three synonym pairs merged (alt → alternating, snap → snap aim, flow → flow aim).
Changing the label space moves every score, so everything the gate reads was
re-measured on the new space rather than carried over. **Section 12's constants
are superseded by this section.**

### The data and the holdout

```
$ python -m mlops.split
[labels] policy kept 22203 of 22901 tag instances; 2 maps now have no labels
  n_rows 4643   n_labels 58   n_holdout 929
  holdout_id_sha256 1da06c95cc6655a63688b0e60ee3061135d9cf80eadfa602be1717e44c0060e4
```

- 603 instances are dropped outright; the other 95 are duplicates removed where a
  map carried both names of a merged pair (alt+alternating 32, snap+snap aim 8,
  flow+flow aim 55).
- The policy runs after the feature cache and never removes rows. The 2 maps
  left without labels keep their rows, so the holdout hash is the one in §2.

### The reference: the shipped models, on the new question

The shipped 66-label ensemble is scored on the 58 labels by projecting its
outputs: dropped columns are discarded, and a merged tag takes the max of its
members (`labels.project_probabilities`). Any other label difference is refused.

```
$ python -c "... score_on_holdout('.', prepared, split) ..."
[labels] Scoring . (66 labels) on the current 58-label space: dropped [comfortable,
  dt speed, fast, practise, progressive difficulty]; merged [alternating <- alt+alternating,
  flow aim <- flow+flow aim, snap aim <- snap+snap aim]
micro_f1=0.5629 macro_f1=0.3723 weighted_f1=0.5425 (never-predicted 1/46 supported, 9/58 overall)
```

`REFERENCE_MICRO_F1` moves from 0.556967 to **0.562950**. Leaving it at the old
value would have let every new-label candidate through about 0.006 below what
users already have, which is over 3 sigma of training noise.

### Noise: ten retrains on the new labels

`python -m tools.calibrate_gate --prefix labels58-seed-` trained the identical
configuration on the 58 labels ten times, varying only `--train-seed`:

```
seed 1   micro_f1=0.556803    seed 6   micro_f1=0.560187
seed 2   micro_f1=0.553828    seed 7   micro_f1=0.557510
seed 3   micro_f1=0.558094    seed 8   micro_f1=0.557664
seed 4   micro_f1=0.558296    seed 9   micro_f1=0.556111
seed 5   micro_f1=0.559237    seed 10  micro_f1=0.553626
mean 0.557136   sigma 0.002130   gap/sigma 3.08
never_predicted_supported: [1, 2, 1, 1, 0, 1, 1, 1, 1, 1]  (champion 1)
```

### Why the champion sits 2.77 sigma above them

The shipped model scores 0.562950, and the retrains average 0.557136. The same
tool, run with `--prefix seed-` over the ten existing **66-label** seeds, scores
those seeds on the 58 labels by projection, which splits the 0.0058 gap in two:

| Comparison | micro F1 | Share of the gap |
|---|---|---|
| shipped champion (projected) | 0.562950 | |
| 66-label seeds, projected (mean of 10, sigma 0.001987) | 0.559025 | 0.0039: the champion was a lucky draw, as §12 already found (1.89 sigma on 66 labels) |
| 58-label retrains (mean of 10) | 0.557136 | 0.0019 (about 2 standard errors): a model trained on a merged tag scores lower at the fixed 0.27 threshold than the max of two separately trained outputs |

### The tolerance, by the documented rule

§12's rule sets k so that a candidate at an ordinary mean − 2 sigma still passes
against the fixed champion: k ≥ (champion − mean)/sigma + 2 = 2.77 + 2 = 4.77, so
**k = 5**, tolerance 5 × 0.002130 = **0.010650**.

```
$ python -m tools.calibrate_gate --prefix labels58-seed- --k 5
  MICRO_F1_SIGMA = 0.002130   TOLERANCE_K = 5   DEFAULT_TOLERANCE = 0.010650
  worst run 0.553626 vs champion floor 0.552300 -> accepted
```

- **Every honest retrain passes** at k = 5. At k = 4 (floor 0.554430) seeds 2
  and 10 would be rejected.
- **The 1-epoch model is still rejected by a wide margin.** Projected onto the
  58 labels it scores 0.420240, against a floor of 0.552300.
- **The never-predicted allowance of +1 still admits every rerun** (0 to 2,
  against the champion's 1).

All three are pinned in `tests/test_gate.py`, including a test that replays the
ten measured seeds and the 1-epoch model through `decide()`.

---

## 15. v2 features: built on the dev split, judged once on the holdout

`features_v2.py` (72 features) replaces v1's measurements where v1 could not see
the pattern a tag names. v1 is untouched (the v1 golden test still passes
bit-for-bit). Everything here uses the 58-label space from §14.

### The holdout was not used to build it

Features were iterated with `tools/feature_probe.py`, which carves a dev split
out of the 3714 **training** rows (2971 train / 743 val, seed 7) and asserts no
holdout row is in it. The 929-map holdout was scored once, at the end.

### Same rows, same holdout

v2 reuses v1's row filter, and `prepare_for_versions` refuses to proceed if the
two versions disagree on which rows exist or in what order:

```
$ python -m mlops.split --feature-version 2
[split] v2 features for 4643 rows; 0 had no map_meta entry and used defaults
  n_features 72   n_labels 58   holdout_id_sha256 1da06c95...   (identical to v1)
```

`cli.py enrich-dataset` found a `.osu` file in `downloads/` for all 4645
dataset entries, with no object-count mismatches.

### Two bugs found on the way, both now guarded

1. **Trick timing points.** Three maps use red lines of 1e-298 ms or more than
   60000 ms. Unclamped, one produced a BPM of 5e301.
   - `StandardScaler` overflowed on it without raising, and the first three v2
     ensembles each learned a constant output (holdout micro F1 0.26, identical
     across seeds).
   - Fixed by clamping beat length to [6, 60000] ms, the range osu! lazer
     allows.
   - `split.scale_all` now **refuses** any column whose standardised values
     exceed √n. That bound is a mathematical limit for a finite column, so
     exceeding it proves an overflow.
2. **Gimmick maps dominating a column.** A 2970-radius slider, a 517-anchor
   slider and a 4830 BPM section each set a column's scale for every map. The
   heavy-tailed features now carry a log: the column's maximum |z| fell from 68.0
   (the √(n−1) limit) to 41.8, against v1's 33.6.

The two tagged failures (1-2 recall, slider jumps) were found and fixed on the
dev split, by comparing which features separate each tag on training rows only:

| Tag | Weak v2 feature | Replacement | Cohen's d |
|---|---|---|---|
| 1-2 | "lands on its start point" (not even in the top 6) | runs of ≥3 direction reversals at any rhythm | **+2.03**, beating v1's best of +1.94 |
| slider jumps | into / out of a slider, pooled | "jump onto a slider" split out | +0.59, vs +0.47 pooled |

### Dev split (single ensemble members, 3 seeds each, final v2)

```
                       v1            v2
micro precision   0.485 ±.002   0.501 ±.008
micro recall      0.625 ±.006   0.637 ±.009
micro F1          0.546 ±.004   0.561 ±.003
false positives    2402 ±11      2300 ±103
```

Ablation, run on v2 before the clamp and log fixes: every group was removed in
turn (3 seeds each). Removing any group lowered micro F1 or left it level, and
raised false positives; the jumps group mattered most (0.560 → 0.519).
`v1+v2` combined (0.556) did not beat v2 alone (0.560), so v1 carries nothing
v2 lacks.

### The holdout: once, at the end

`python -m tools.tag_quality --arm champion . --arm v1-58labels <10 seeds> --arm v2 <3 seeds>`

The v1 arm is ten v1-feature ensembles retrained on the same 58 labels, so the
v1-vs-v2 difference is the features alone.

```
                  champion   v1-58labels (10)   v2 (3)
micro precision     0.507     0.499 ±.003       0.522 ±.004
micro recall        0.633     0.630 ±.005       0.658 ±.002
micro F1            0.563     0.557 ±.002       0.582 ±.003
false positives      2778      2847 ±42          2716 ±36
```

**Micro F1 is +0.025 over the v1 retrains**, about 12 sigma of seed noise.
Precision and recall rise together.

Per-tag precision at 0.27 (v1 retrains → v2):

| Change | Tags |
|---|---|
| Improved | **1-2 0.39 → 0.51**, **cut streams 0.26 → 0.46**, streams 0.55 → 0.62, square jumps 0.78 → 0.93, sharp angles 0.51 → 0.56, variable streams 0.36 → 0.42, large jumps 0.49 → 0.53, high spacing 0.44 → 0.47, snap aim 0.31 → 0.35, triples, bursts, burst sliders; **variable bpm** is predicted at all (precision 0.75, from never) |
| Flat | jumps, short jumps, flow aim, slider jumps, alternating |
| Worse | doubles 0.36 → 0.33 (recall unchanged) |

Ranking ability (AUC) improved for:
- snap aim 0.796 → 0.845;
- cut streams 0.898 → 0.942;
- square jumps 0.955 → 0.983;
- linear aim 0.787 → 0.828;
- variable bpm 0.666 → 0.794.

Sibling-tag coupling fell where v2 added the distinguishing feature (cut streams
/ streams 0.78 → 0.65; spaced streams / streams 0.84 → 0.77). For most pairs it
is still well above the label correlation.

**Buzz sliders.** Precision went to 0 at the threshold. Its AUC is 0.842 → 0.811
on 5 holdout maps, which is noise-sized. The v2 feature separates the tag as
well as v1's does on training rows (d +2.87 vs +2.95).

### The gate, across feature versions

The runs used a throwaway registry (`MLFLOW_TRACKING_URI=sqlite:///.tmp_mlflow_v2gate.db`)
and temp root directories:

```
promote --candidate .                     -> PROMOTE (first champion, 0.562950)
promote --candidate candidates/v2-seed-1  -> PROMOTE
  Candidate: micro_f1=0.5792 (v2 features)
  Champion re-scored on this split: micro_f1=0.5629 (v1 features, label projection)
  [PASS] micro_f1 vs champion   0.579220 vs floor 0.552300
  [PASS] micro_f1 vs reference  0.579220 vs floor 0.552300
  [PASS] never-predicted tags   0 vs limit 2
promote --candidate candidates/v2-seed-2  -> PROMOTE
  Champion re-scored: micro_f1=0.5792   <- the v2 champion came back out of the
                                           registry and was fed v2 features
export-onnx --model-dir candidates/v2-seed-1
  -> refused, exit 1: "holds a v2-feature model, but the app computes v1 features"
```

The real `mlflow.db` and the repo-root models were not modified.

### By eye, on `songs/`

`python -m tools.compare_on_maps --model-dir . --model-dir candidates/v2-seed-1`
covers the 31 committed maps. 12 of them are in the dataset, so their community
tags serve as a reference:

| Where the two models disagree | Community tags among them |
|---|---|
| tags only the shipped model gives | **2 of 30** |
| tags only v2 gives | 5 of 12 |

- v2 is closer to the community tags on 8 of the 12 maps and further on none.
- **On the 5 of those maps that are in the holdout** (unseen by both models), v2
  is closer on 4 and ties on 1.

---

## 16. Threshold 0.26, redundant tags, and the C# port

Added 2026-09-30.

### The decision rule

- **Threshold 0.26**, decided from the dev-split sweep (`tools/feature_probe.py
  --members 5 --sweep`, training rows only). Micro F1 is flat from about 0.26
  to 0.34 and falls either side.
- **Applied at display precision.** A tag counts when `p >= 0.26 - 0.005`, so a
  tag shown as 0.26 is predicted. `tests/test_labels.py` pins the boundary:
  0.2551 is in and 0.2549 is out.
- **Written into `model_config.json`** (`threshold`, `display_decimals`), so the
  app reads it instead of compiling it in.

### The gate, re-measured at 0.26

Same method as §14: scoring only, over the same ten 58-label seeds.

```
$ python -m tools.calibrate_gate --prefix labels58-seed- --k 5
seeds: 0.555957 0.552091 0.555059 0.557399 0.557573 0.559607 0.554348 0.555387 0.556137 0.552354
mean 0.555591   sigma 0.002318   champion (projected) 0.561577   never(supported) 0..1
worst run 0.552091 vs champion floor 0.549987 -> accepted
```

- (0.561577 − 0.555591) / 0.002318 = **2.58** sigma, plus 2 gives 4.58, so
  **k stays 5**.
- Tolerance is 0.011590 and `REFERENCE_MICRO_F1` is 0.561577293075531.
- The 1-epoch model scores 0.414121 and is rejected.
- The v2 candidates score 0.577803, 0.583103 and 0.576743.

### Redundant tags

`labels.SUPPRESSED_BY`, measured on v2's holdout predictions:

| Hidden tag | When shown next to | Evidence |
|---|---|---|
| jumps | large / short / cross screen jumps | predicted with large jumps 57% of the time (community 29%), with short jumps 86% (community 53%) |
| high spacing | large / cross screen jumps | 65% of its predictions come with large jumps; alone it is right 33% of the time. The community uses it independently on 242 of 445 maps, so it is hidden rather than dropped |

- Hiding `jumps` moves displayed micro precision 0.509 → 0.510 and recall
  0.677 → 0.668.
- It is presentation only: the gate still scores raw predictions.
- The app searches by substring, so a `jumps` search still finds these maps.

### C# port: feature parity

`FeatureExtractorV2.cs`, run through `parity/ParityDump --feature-version 2`:

| Test | Result |
|---|---|
| Golden map (Polyphia) | all 72 features within 3e-14 |
| App fixtures stream / jump / tech (v2) | PASS at atol = rtol = 1e-9 |
| App fixtures stream / jump / tech (v1) | still PASS: the parser change did not move v1 |
| All 4974 files in `downloads/` (`--dir` mode against Python) | every one of the 4961 readable maps within 9.16e-13 (median per-map worst 1.8e-14) |
| `songs/` (31 maps) | all within 7e-14 |

The 13 remaining files are:
- **12 zero-byte files** (failed downloads);
- **one 14.6 MB file.** The app's parser deliberately refuses files over 5 MB.

The app skips both kinds before extracting any features.

**One real disagreement, and it was a feature flaw.**
- On `downloaded_215463.osu`, `aim_trend` came out +0.047 in Python and −0.017
  in C#.
- Every 4-second window there has the same aim load (0.21333…), so the "trend"
  correlated floating-point rounding (1e-16) with time, and summation order
  decided its sign.
- **Fix:** a load series whose spread is under 1e-9 of its size counts as flat
  (trend 0), in both languages. It affects exactly one of the 4626 dataset maps;
  the next flattest has a relative spread of 4e-5.
- The v2 candidates were not retrained for one feature on one map.
- The v2 golden is unchanged, and a new test covers a perfectly regular map.

### C# port: end to end

The pieces:
- `candidates/v2-seed-2` was exported with
  `cli.py export-onnx --allow-feature-version 2`; without the flag the export
  refuses.
- A scratch harness compiled the app's real `OsuParser`, `FeatureExtractorV2`
  and `OsuClassifier` against those ONNX files and ran them on the 31 `songs/`
  maps.

The results:
- **Probabilities** are within 2.1e-7 of Python's Keras ensemble (float32).
- **Final tag lists**, after the 0.26 rule and redundancy suppression, are
  identical on all 31 maps.
- **The app builds** (`dotnet build`) with no new warnings.

---

## 17. The package layout

Added 2026-10-01. Library code moved into an `osu_tagger/` package. Sections
above name files by their old paths; this is the map:

| Was | Now |
|---|---|
| `osu_parser.py` | `osu_tagger/parsing.py` |
| `neural_model.py` | `osu_tagger/features/v1.py`, plus a root `neural_model.py` shim so the legacy pickle loads |
| `features_v2.py` | `osu_tagger/features/v2.py` |
| `echosu_api.py`, `tag_scraper.py`, `dataset_builder.py`, `rebuild_from_downloaded.py`, `map_meta.py` | `osu_tagger/data/` (`echosu`, `tags`, `builder`, `rebuild`, `map_meta`) |
| `complete_pipeline.py` | `osu_tagger/data/osu_api.py`: only the osu! API login and download, which build-dataset uses |
| `ensemble_evaluator.py` | `osu_tagger/training/ensemble.py` |
| `export_to_onnx.py`, `extract_config.py` | `osu_tagger/export/onnx.py`, `osu_tagger/export/config.py` |
| `parity_dump.py`, `make_goldens.py` | `python -m osu_tagger.parity.dump`, `python -m osu_tagger.parity.goldens` |

**Removed:**
- `retrain_model.py` (legacy single-model corrections loop);
- `tag_fingerprint.py` (one-off analysis);
- `predict_for_overlay.py` (only the retired OsuScout app called it);
- the single-map demo in `complete_pipeline.py`, which called an Echo API
  method that no longer exists.

**Verified after the move:**
- `git mv`, so history follows each file.
- All 109 tests pass and ruff is clean.
- All 27 modules import, every `cli.py` subcommand's `--help` runs, and the
  legacy `beatmap_classifier.pkl` still loads through the shim.
- `python -m osu_tagger.parity.dump --feature-version 2` reproduces the v2
  golden exactly (max diff 0.0).
- `cli.py evaluate --holdout` scores the promoted v2 model end to end.

---

## Caveats

These qualify the numbers above. None of them are defects introduced by this
work; the first two are pre-existing properties of the pipeline that were left
in place deliberately.

1. **The scaler is fit on all 4643 rows before splitting**
   (`ensemble_evaluator.py`, preserved in `mlops/split.py:scale_all`). The holdout's
   feature distribution therefore leaks into standardisation, making every
   absolute score mildly optimistic. Fixing it would change `scaler_mean` /
   `scaler_scale`, hence `model_config.json`, hence C# parity — explicitly out
   of scope. It applies identically to every model the gate compares, so the
   comparison remains fair even though the absolute value is flattering.

2. **Gate metrics score raw thresholded output.** The predict path applies an
   expert-system override that forces the `streams` tag on maps with a 15+ note
   sequence; the gate does not. It measures the network, matching the existing
   `classification_report`, and is not a measurement of end-to-end app
   behaviour.

3. **`songs/` contains 31 committed `.osu` files.** These predate this work. No
   further beatmap content was added — `samples/sample_features.csv` holds
   derived statistics and tag labels only.

---

## Not done

Deliberately out of scope. None of this is implemented, and nothing above
implies otherwise.

- **No cloud anything.** Tracking is a local SQLite file and a local `mlruns/`
  directory. No remote tracking server, no cloud storage, no hosted registry.
- **No scheduled deployment.** Nothing runs on a timer or a trigger. Every
  command in this pipeline is run by hand.
- **No production serving.** The models are consumed by a separate desktop app
  that loads ONNX files off disk. There is no inference server, no API, no
  container serving predictions.
- **No automatic retraining trigger.** The drift report is a batch check a human
  runs and reads. Nothing acts on it — a drifted distribution does not start a
  retrain, and is not wired to any alert.
- **No CD to the app repo.** Promotion updates the local registry and the repo
  root. Copying the 6 files into `OsuScoutNew/Assets/` and cutting a release
  remains a manual step, in a repository this work does not touch.
- **v2 is not yet released in the app.** The port exists and is verified (§16)
  on the app repo's `feat/v2-features` branch, but no v2 model has been promoted
  in the real registry, copied into `OsuScoutNew/Assets/`, or released.
