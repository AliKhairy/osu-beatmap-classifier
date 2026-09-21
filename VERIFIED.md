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
################ (d) EMPTY REGISTRY -> first model promotes ################
Candidate: macro_f1=0.3551 ...
PROMOTE: no champion registered, promoting candidate (macro F1 0.3551) as the first one
EXIT CODE: 0

################ (b) CANDIDATE vs CHAMPION -> promotes ################
Candidate: macro_f1=0.3504 ...
Champion: registered version 2 (run c7f9c287), stored macro_f1=0.3550680494867983
Champion re-scored on this split: macro_f1=0.3551 ...
PROMOTE: macro F1 0.3504 is within tolerance of the champion 0.3551 (delta -0.0047, tolerance 0.0080)
EXIT CODE: 0

################ (c) 1-EPOCH CANDIDATE -> REJECTED ################
Candidate: macro_f1=0.1745 micro_f1=0.4140 ...
Champion re-scored on this split: macro_f1=0.3504 ...
REJECT: macro F1 0.1745 is worse than the champion 0.3504 by more than the tolerance 0.0080 (floor 0.3424)
EXIT CODE: 2
(c) temp root UNCHANGED after rejection - rejected candidate did not overwrite anything
```

**Result: pass.** The weak candidate (`--epochs 1`, macro F1 0.1745 vs the
champion's 0.3504) is rejected with a non-zero exit, and nothing is copied.

Note the `Champion re-scored on this split` line: the gate does not trust the
champion's stored metric, it re-runs it through the same scoring path as the
candidate. In (b) the stored 0.35507 and the re-scored 0.3551 agree, which is
what you want to see — but if the dataset had moved, they would not, and the
comparison would have been against a number describing a different test set.

---

## 6. The shipped models were never touched

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
