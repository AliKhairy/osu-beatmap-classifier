import tensorflow as tf
import os
import subprocess
import shutil
import sys

def convert_models():
    for i in range(1, 6):
        model_name = f"ensemble_model_{i}.keras"
        onnx_name = f"ensemble_model_{i}.onnx"
        temp_dir = f"temp_savedmodel_{i}"
        
        if not os.path.exists(model_name):
            print(f"Skipping: Could not find {model_name}")
            continue
            
        print(f"Loading {model_name} with Keras 3...")
        model = tf.keras.models.load_model(model_name, compile=False)
        
        print(f"Exporting {model_name} to raw SavedModel format...")
        model.export(temp_dir)
        
        print("Running tf2onnx conversion...")
        
        # THE FIX: sys.executable forces the subprocess to stay inside the venv
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", temp_dir,
            "--output", onnx_name,
            "--opset", "15"
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Successfully saved {onnx_name}")
        
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
        print("-" * 40)

if __name__ == "__main__":
    convert_models()