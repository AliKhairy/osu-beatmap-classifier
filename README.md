# osu! Beatmap Classifier

## Overview

`osu-beatmap-classifier` is a machine learning project designed to analyze `.osu` beatmap files and predict descriptive tags such as "streams," "jumps," and "finger control." It uses a neural network trained on data scraped from [echosu.com](https://echosu.com/) to learn the relationship between hit object patterns and common mapping terminology.

This tool can be used to automatically tag a library of beatmaps, assist mappers in understanding their creations, or serve as a foundation for more advanced beatmap analysis tools.

## Features

-   **Data Collection**: Builds a dataset by downloading beatmap info and tags from the Echo API.
-   **Advanced Feature Extraction**: Analyzes hit object data to extract 90 meaningful geometric features, including stream purity, finger control metrics, and global snap variance.
-   **Deep Learning Architecture**: Uses a TensorFlow/Keras Dense Neural Network to classify beatmaps into multiple overlapping tag categories.
-   **5-Model Ensemble Learning**: Features a robust voting classifier that trains 5 distinct neural networks simultaneously, reducing variance and correcting single-model bias on subjective tags.
-   **Deterministic Feature Injection**: Hard-coded mechanical rules (e.g., forcing the "streams" tag if a 15+ note sequence is detected) to prevent the black-box AI from missing absolute geometric truths.
-   **Interactive CLI**: A command-line interface to easily train models, evaluate ensembles, and predict tags for local `.osu` files.

## How It Works

The project follows a standard, modular machine learning pipeline:
1.  **Dataset Construction** (`dataset_builder.py`, `rebuild_from_downloaded.py`): Beatmap IDs and tags are fetched from the Echo API. The corresponding `.osu` files are downloaded.
2.  **Parsing & Feature Extraction** (`osu_parser.py`, `neural_model.py`): The `.osu` files are parsed to extract raw coordinates. This is transformed into a high-dimensional feature vector representing micro-patterns and spatial flow.
3.  **Model Training** (`neural_model.py`): The feature vectors train a multi-label classification neural network. The trained baseline model is saved to `beatmap_classifier.pkl`.
4.  **Ensemble Evaluation** (`ensemble_evaluator.py`): Trains 5 independent models (`ensemble_model_1.keras` to `5`) and aggregates their probabilities to drastically improve F1-Scores on minority tags.
5.  **Prediction** (`main.py`, `predict_for_overlay.py`): Predicts tags for any new `.osu` file, automatically prioritizing the 5-model ensemble if it detects one on disk.

## Setup and Installation

**Prerequisites:**
-   Python 3.8+
-   Git

**1. Clone the repository:**
```bash
git clone https://github.com/AliKhairy/osu-beatmap-classifier.git
cd osu-beatmap-classifier
```

**2. Set up a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Configure your API Token:**
You need an API token from `echosu.com`.
-   Create a file named `.env` in the root of the project directory.
-   Add your token to this file like so:
    ```
    ECHO_API_TOKEN="YourEchosuApiTokenHere"
    OSU_CLIENT_ID = 'YourID'
    OSU_CLIENT_SECRET = 'YourOsuApiTokenHere'
    ```

**5. Prepare Beatmap Folders:**
The application uses two folders for `.osu` files:
-   `downloads/`: Used by `rebuild_from_downloaded.py` to process local maps.
-   `songs/`: Used by the interactive prediction menu in `main.py` to find maps for testing.

Create these folders if they don't exist and place some `.osu` files inside them.

## Usage

The main entry point is `main.py`. You can run it from the command line:

```bash
python main.py
```

This will launch an interactive menu that guides you through the following options:
-   **Build Dataset & Train:** If no model exists, it will automatically start the process.
-   **Predict Tags**: Analyze a single `.osu` file from your `songs` folder.
-   **Test Model**: Run predictions on multiple maps to see the model's performance.
-   **Retrain/Rebuild**: Update the model or rebuild the entire dataset from scratch.

## Exporting Models for the App (ONNX)

The desktop app (**OsuScoutNew**) does not run Python or Keras. It runs inference
on-device using ONNX Runtime. Training produces Keras/scikit-learn artifacts; two
scripts convert those into the exact files the app consumes:

| Training artifact                     | Export script         | App file (`OsuScoutNew/Assets/`) |
| ------------------------------------- | --------------------- | -------------------------------- |
| `ensemble_model_1..5.keras`           | `export_to_onnx.py`   | `ensemble_model_1..5.onnx`       |
| `ensemble_scaler.pkl` + `..._binarizer.pkl` | `extract_config.py` | `model_config.json`              |

**To regenerate the model files after (re)training:**
```bash
python export_to_onnx.py   # .keras -> .onnx (tf2onnx, opset 15)
python extract_config.py   # scaler + binarizer -> model_config.json
```
This produces 6 files: `ensemble_model_1.onnx` ... `ensemble_model_5.onnx` and
`model_config.json`.

## Shipping a Model Update to the App

The app auto-updates via Velopack/GitHub Releases, and the model files are bundled
into the app (marked `CopyToOutputDirectory` in `OsuScoutNew.csproj`). So shipping a
new model is the same as shipping any app update:

1. **Retrain** here (`python main.py` -> retrain, or `retrain_model.py`).
2. **Export** the 6 files (see above).
3. **Copy** all 6 into `OsuScoutNew/Assets/`, replacing the old ones.
4. In the app repo: bump the version, `dotnet publish`, `vpk pack`, and upload the
   release. Users auto-update and receive the new model on next launch.

> **Feature count must stay in sync.** The C# `FeatureExtractor` computes the input
> vector (currently 90 features) and `OsuClassifier` validates that exact length. If
> you change the **number or order of features** in `neural_model.py`, you must make
> the **identical** change in the app's `FeatureExtractor.cs` and retrain. Adding new
> **tags** (without changing feature count) needs no C# change — the tag list is read
> from `model_config.json` at runtime.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
