# osu! Beatmap Classifier

## Overview

`osu-beatmap-classifier` is a machine learning project designed to analyze `.osu` beatmap files and predict descriptive tags such as "streams," "jumps," and "finger control." It uses a neural network trained on data scraped from [echosu.com](https://echosu.com/) to learn the relationship between hit object patterns and common mapping terminology.

This tool can be used to automatically tag a library of beatmaps, assist mappers in understanding their creations, or serve as a foundation for more advanced beatmap analysis tools.

## Features

-   **Data Collection**: Builds a dataset by downloading beatmap info and tags from the Echo API.
-   **Local Processing**: Can also process a local folder of `.osu` files to build or supplement a dataset.
-   **Advanced Feature Extraction**: Analyzes hit object data to extract meaningful features like stream scores, finger control metrics, and pattern instability.
-   **Neural Network Model**: Uses a TensorFlow/Keras model to classify beatmaps into multiple tag categories.
-   **Model Training & Retraining**: Full pipeline for training, saving, and retraining the model with new or corrected data.
-   **Interactive CLI**: A command-line interface to easily train the model, test it, and predict tags for new maps.

## How It Works

The project follows a standard machine learning pipeline:
1.  **Dataset Construction** (`dataset_builder.py`, `rebuild_from_downloaded.py`): Beatmap IDs and tags are fetched from the Echo API. The corresponding `.osu` files are downloaded.
2.  **Parsing & Feature Extraction** (`osu_parser.py`, `neural_model.py`): The `.osu` files are parsed to extract hit object data (positions, times, types). This raw data is then transformed into a high-dimensional feature vector representing musical and positional patterns.
3.  **Model Training** (`neural_model.py`): The feature vectors and their corresponding tags are used to train a multi-label classification neural network. The trained model (including the data scaler and label binarizer) is saved to `beatmap_classifier.pkl`.
4.  **Prediction** (`main.py`, `predict_for_overlay.py`): The trained model can be loaded to predict tags for any new `.osu` file.

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
    ECHO_API_TOKEN="YourApiTokenHere"
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

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
