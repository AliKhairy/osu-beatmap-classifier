"""
The osu! beatmap tagger: everything between a .osu file and a tag prediction.

    parsing     reading .osu files (hit objects, difficulty, timing points)
    features    the model's input vectors: v1 (90, what the shipped app has
                computed so far) and v2 (72, see docs/feature_v2_spec.md)
    data        building the dataset: the Echo tag API, the osu! API, the
                tag scraper, and the map_meta.json sidecar
    training    training the 5-model ensemble
    export      turning trained models into the ONNX files + model_config.json
                the desktop app loads
    parity      dumping feature vectors to check the C# port against

Measuring and gating models lives next door in mlops/; one-off analysis
tools in tools/. The command-line entry point is cli.py at the repo root.
"""
