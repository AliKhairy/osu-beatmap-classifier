"""
Experiment tracking, evaluation and the promotion gate.

Grouped as a package because these seven modules are one concern - deciding
whether a model is good enough to ship - and they were cluttering the repo root
alongside the training code.

What deliberately stays at the root, and why:

  neural_model.py    beatmap_classifier.pkl pickles an ImprovedBeatmapClassifier
                     INSTANCE, and pickle stores the module path. Moving this
                     file makes that artifact unloadable with
                     ModuleNotFoundError: No module named 'neural_model' -
                     at runtime, not import time.
  cli.py             the Dockerfile ENTRYPOINT is ["python", "cli.py"].
  parity_dump.py     invoked by the app repo's parity harness
  make_goldens.py    (OsuScoutNew/parity/compare_parity.py), which this work
                     is not allowed to touch.

Modules here import the root-level training code (neural_model, osu_parser,
ensemble_evaluator) normally; the repo root is on sys.path whenever cli.py or
main.py runs.
"""
