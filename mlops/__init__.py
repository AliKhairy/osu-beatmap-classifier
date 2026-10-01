"""
Experiment tracking, evaluation and the promotion gate.

Grouped as a package because these modules are one concern - deciding whether
a model is good enough to ship - separate from the code that builds and runs
models, which is the osu_tagger package.

What stays at the repo root, and why:

  cli.py             the Dockerfile ENTRYPOINT is ["python", "cli.py"].
  main.py            the interactive menu.
  neural_model.py    a shim. beatmap_classifier.pkl pickles an
                     ImprovedBeatmapClassifier INSTANCE, and pickle stores the
                     module path; without a module of that name the file fails
                     to load with ModuleNotFoundError at runtime. The code
                     itself is osu_tagger.features.v1.

The app repo's parity harness (OsuScoutNew/parity) calls the Python side as
`python -m osu_tagger.parity.dump` and `python -m osu_tagger.parity.goldens`.
"""
