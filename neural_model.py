"""
Compatibility shim: the v1 feature extractor now lives in osu_tagger.features.v1.

This file stays at the repo root for one reason. beatmap_classifier.pkl (the
legacy single model) pickles an ImprovedBeatmapClassifier INSTANCE, and pickle
records the class by module path - 'neural_model'. Without a module of that
name, loading it fails with ModuleNotFoundError at runtime. Import from
osu_tagger.features.v1 in new code.
"""
from osu_tagger.features.v1 import *  # noqa: F401,F403
from osu_tagger.features.v1 import ImprovedBeatmapClassifier  # noqa: F401
