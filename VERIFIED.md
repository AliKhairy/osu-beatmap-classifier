# VERIFIED

Every claim below was produced by running the stated command on this machine.
Anything not actually run is in [Not done](#not-done) or [Caveats](#caveats) —
not asserted anywhere else.

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
$ python split.py                      # run 1, cold (no feature cache)
$ python split.py                      # run 2, warm cache
$ python split.py --no-cache           # run 3, full re-extraction from raw JSON
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
$ python -c "from tracking import describe_run; ..."
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

## 4. The gate tolerance was measured, not chosen

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
`promote.py:DEFAULT_TOLERANCE`.

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

Raw numbers: `seed_calibration.json`.

---

## 5. Registry and promotion gate

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
$ python -m pytest tests/ -q
.....................................                                    [100%]
37 passed

$ python -m ruff check .
All checks passed!
```

`tests/test_gate.py` (16 tests) covers `promote.decide` — better, worse within
tolerance, worse beyond tolerance, the inclusive boundary at exactly
`champion - tolerance`, no champion, zero tolerance, a negative tolerance being
rejected, a missing tolerance refusing to guess, and the ratchet case: a
candidate that beats the incumbent but sits below `best_ever - tolerance` must
be rejected.

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
wrote to a temp directory via `--root-dir`. **Promoting a candidate into the
real repo root has not been done and is left as an explicit decision**, to be
made after reviewing the numbers above.

---

## Caveats

These qualify the numbers above. None of them are defects introduced by this
work; the first two are pre-existing properties of the pipeline that were left
in place deliberately.

1. **The scaler is fit on all 4643 rows before splitting**
   (`ensemble_evaluator.py`, preserved in `split.py:scale_all`). The holdout's
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

## Not verified

One thing in this branch was **not** verified locally, stated here rather than
implied to work:

- **The Docker image build.** The Dockerfile now installs `requirements.txt`
  and `requirements-mlops.txt` as two separate layers. Docker Desktop's daemon
  was not running on this machine (`failed to connect to the docker API at
  npipe:////./pipe/dockerDesktopLinuxEngine`), so the image was never built
  here. What *was* checked is only static: both `COPY` sources exist and the
  ignore rules are in place. CI builds the image on every push and runs the
  `--help`, failure-exit and image-contents checks against it, so this is
  covered there — but it has not been run locally.

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
- **No BPM feature.** Known to be missing from the 90-feature vector; changing
  the vector would break C# parity and was excluded from this work.
