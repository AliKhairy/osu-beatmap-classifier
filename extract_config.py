import json
import os
import pickle


def extract_config(model_dir='.', out_path=None):
    """
    Turn the fitted scaler and binarizer into model_config.json for the C# app.

    The app reads scaler_mean, scaler_scale and the tag list from this file at
    runtime, so it MUST be regenerated from the same artifacts that produced the
    .onnx files. Shipping a model_config.json from a different training run
    silently corrupts every prediction: the vector is standardised by the wrong
    constants and nothing crashes.

    That coupling is why cli.py's export-onnx now calls this as part of the same
    step instead of leaving it as a separate command to remember.

    Returns the path written, or None if the inputs were missing.
    """
    scaler_path = os.path.join(model_dir, "ensemble_scaler.pkl")
    binarizer_path = os.path.join(model_dir, "ensemble_binarizer.pkl")
    out_path = out_path or os.path.join(model_dir, "model_config.json")

    if not os.path.exists(scaler_path) or not os.path.exists(binarizer_path):
        print(f"Error: Could not find the .pkl files in {model_dir}.")
        return None

    print("Loading PKL files...")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(binarizer_path, "rb") as f:
        binarizer = pickle.load(f)

    print("Extracting math and tags...")

    from mlops.labels import DISPLAY_DECIMALS, SUPPRESSED_BY, THRESHOLD
    from mlops.split import model_feature_version

    # Scikit-learn stores these as numpy arrays, we convert to standard Python lists.
    #
    # Beyond the scaler and tags, the app reads its decision rule from here rather
    # than compiling it in: which extractor to run (feature_version), the cutoff
    # (threshold, applied at display precision so a tag shown as 0.26 counts at
    # 0.26), and which general tags to hide beside specific ones. A retune then
    # ships as a new config file, not an app code change.
    config = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "tags": binarizer.classes_.tolist(),
        "feature_version": model_feature_version(model_dir),
        "threshold": THRESHOLD,
        "display_decimals": DISPLAY_DECIMALS,
        "suppressed_by": {tag: list(by) for tag, by in SUPPRESSED_BY.items()},
    }

    print(f"Saving to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=4)

    print("Success! The C# bridge is ready.")
    return out_path


if __name__ == "__main__":
    extract_config()
