import os
import shutil
import subprocess
import sys

import tensorflow as tf


def convert_models(model_dir='.', out_dir=None, num_models=5):
    """
    Convert ensemble_model_N.keras to ONNX for the desktop app.

    Paths are parameters now because models no longer only ever live in the repo
    root - a candidate trains into candidates/<name>/ and may be exported from
    there. The defaults reproduce the original behaviour exactly.

    Returns the list of .onnx files actually written, so the caller can tell a
    partial export from a complete one. The original returned nothing and merely
    printed a warning per missing model, which is how a broken half-export could
    exit 0.
    """
    out_dir = out_dir or model_dir
    os.makedirs(out_dir, exist_ok=True)
    written = []

    for i in range(1, num_models + 1):
        model_name = os.path.join(model_dir, f"ensemble_model_{i}.keras")
        onnx_name = os.path.join(out_dir, f"ensemble_model_{i}.onnx")
        temp_dir = os.path.join(out_dir, f"temp_savedmodel_{i}")

        if not os.path.exists(model_name):
            print(f"Skipping: Could not find {model_name}")
            continue

        print(f"Loading {model_name} with Keras 3...")
        model = tf.keras.models.load_model(model_name, compile=False)

        print(f"Exporting {model_name} to raw SavedModel format...")
        model.export(temp_dir)

        print("Running tf2onnx conversion...")

        # sys.executable forces the subprocess to stay inside the venv
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", temp_dir,
            "--output", onnx_name,
            "--opset", "15"
        ]

        subprocess.run(cmd, check=True)
        print(f"Successfully saved {onnx_name}")
        written.append(onnx_name)

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        print("-" * 40)

    return written


if __name__ == "__main__":
    convert_models()
