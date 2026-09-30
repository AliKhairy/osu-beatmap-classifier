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

    # Scikit-learn stores these as numpy arrays, we convert to standard Python lists
    config = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "tags": binarizer.classes_.tolist()
    }

    print(f"Saving to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=4)

    print("Success! The C# bridge is ready.")
    return out_path


if __name__ == "__main__":
    extract_config()
