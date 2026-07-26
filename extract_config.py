import pickle
import json
import os

def extract_config():
    scaler_path = "ensemble_scaler.pkl"
    binarizer_path = "ensemble_binarizer.pkl"
    
    if not os.path.exists(scaler_path) or not os.path.exists(binarizer_path):
        print("Error: Could not find the .pkl files.")
        return

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
    
    print("Saving to model_config.json...")
    with open("model_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("Success! The C# bridge is ready.")

if __name__ == "__main__":
    extract_config()