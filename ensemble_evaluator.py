import os
import json
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

from neural_model import ImprovedBeatmapClassifier
from osu_parser import OsuFileParser

def train_and_evaluate_ensemble(num_models=5):
    print(f"--- Starting {num_models}-Model Ensemble Training ---")
    classifier = ImprovedBeatmapClassifier()
    
    try:
        with open('ml_dataset.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("Error: ml_dataset.json not found.")
        return

    X_raw, y_raw = [], []
    for sample in dataset:
        if sample.get('tags') and sample.get('hit_objects'):
            sections = classifier.split_beatmap_into_sections(sample['hit_objects'])
            if not sections: continue
            vec = classifier._aggregate_features_for_map(sections)
            if vec is not None:
                X_raw.append(vec)
                y_raw.append(sample['tags'])

    X = np.array(X_raw)
    
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    label_binarizer = MultiLabelBinarizer()
    y_binary = label_binarizer.fit_transform(y_raw)
    classes = label_binarizer.classes_
    
    # --- NEW: Save the Ensemble Scaler and Binarizer ---
    with open('ensemble_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('ensemble_binarizer.pkl', 'wb') as f:
        pickle.dump(label_binarizer, f)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)
    
    trained_models = []
    input_shape = X_train.shape[1]
    output_shape = y_train.shape[1]
    
    for i in range(num_models):
        print(f"\n>>> Training Model {i+1}/{num_models} <<<")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(output_shape, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
        
        # --- NEW: Save each model to disk ---
        model_filename = f'ensemble_model_{i+1}.keras'
        model.save(model_filename)
        trained_models.append(model)
        print(f"Model {i+1} saved as {model_filename}.")

    print("\n--- Generating Ensemble Predictions ---")
    all_predictions = [model.predict(X_test, verbose=0) for model in trained_models]
    averaged_probabilities = np.mean(all_predictions, axis=0)
    
    final_binary_predictions = (averaged_probabilities >= 0.27).astype(int)

    print("\n" + "="*50)
    print(f"ENSEMBLE ({num_models} MODELS) CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, final_binary_predictions, target_names=classes, zero_division=0))


def load_ensemble_assets(num_models=5):
    """Loads the scaler, binarizer, and all 5 models into RAM once."""
    if not os.path.exists('ensemble_scaler.pkl') or not os.path.exists('ensemble_model_1.keras'):
        return None, None, None

    print(f"\n[System] Loading {num_models} Neural Networks into memory...")
    with open('ensemble_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('ensemble_binarizer.pkl', 'rb') as f:
        label_binarizer = pickle.load(f)

    models = []
    for i in range(num_models):
        models.append(tf.keras.models.load_model(f'ensemble_model_{i+1}.keras'))
        
    return scaler, label_binarizer, models


def predict_with_ensemble(osu_file_path, threshold, assets, classifier):
    """Predicts tags for a single map using pre-loaded ensemble assets."""
    scaler, label_binarizer, models = assets
    
    parser = OsuFileParser(osu_file_path)
    parser.read_file()
    metadata = parser.get_metadata()
    print(f"\nPredicting for: {metadata.get('Artist')} - {metadata.get('Title')} [{metadata.get('Version')}]")

    sections = classifier.split_beatmap_into_sections(parser.extract_raw_hit_objects())
    if not sections: return ["Map is too short."]

    raw_features = classifier._aggregate_features_for_map(sections)
    if raw_features is None: return ["Feature extraction failed."]

    X_scaled = scaler.transform(raw_features.reshape(1, -1))

    # Get predictions from all loaded models
    predictions = [model.predict(X_scaled, verbose=0) for model in models]
    avg_probs = np.mean(predictions, axis=0)[0]

    predicted_tags = []
    classes = label_binarizer.classes_
    
    print("Prediction Probabilities:")
    for i, tag in enumerate(classes):
        prob = avg_probs[i]
        is_predicted = prob >= threshold
        # Only print the positive hits to keep the terminal clean for multi-map
        if is_predicted:
            print(f"  [x] {tag:<20} | Probability: {prob:.3f}")
            predicted_tags.append(tag)

    # --- Expert System Post-Processing ---
    max_stream_length = raw_features[2]
    alt_prob = avg_probs[list(classes).index('alternating')] if 'alternating' in classes else 0.0

    if max_stream_length >= 15 and alt_prob < 0.35:
        if 'streams' not in predicted_tags and 'streams' in classes:
            predicted_tags.append('streams')
            print(f"  [!] OVERRIDE: {int(max_stream_length)}-note stream detected. Forcing 'streams' tag.")

    predicted_tags.sort()
    return predicted_tags if predicted_tags else ["No tags above threshold."]


def test_multiple_maps_with_ensemble(max_maps=5, threshold=0.27):
    """Tests the ensemble on a batch of local maps instantly."""
    songs_folder = "songs"
    if not os.path.exists(songs_folder):
        print(f"Error: Songs folder not found at '{songs_folder}'")
        return

    osu_files = [f for f in os.listdir(songs_folder) if f.endswith('.osu')]
    if not osu_files:
        print(f"Error: No .osu files found in '{songs_folder}'")
        return

    assets = load_ensemble_assets()
    if assets[0] is None:
        print("Error: Ensemble not trained. Run Option 2 first.")
        return

    classifier = ImprovedBeatmapClassifier()
    maps_to_test = osu_files[:max_maps]
    
    print(f"\n{'='*50}\nENSEMBLE BATCH TESTING ({len(maps_to_test)} MAPS)\n{'='*50}")
    
    for file in maps_to_test:
        map_path = os.path.join(songs_folder, file)
        tags = predict_with_ensemble(map_path, threshold, assets, classifier)
        print(f"Final Tags: {tags}\n{'-'*40}")