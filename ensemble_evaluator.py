import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from neural_model import ImprovedBeatmapClassifier
from osu_parser import OsuFileParser

# The confidence cutoff, named rather than scattered as a literal. This value is
# baked into the deployed C# app's expectations and into every recorded metric,
# so it is a constant to be read, not a parameter to be tuned here. Changing it
# invalidates the numbers in the model registry.
THRESHOLD = 0.27

def train_and_evaluate_ensemble(num_models=5, dataset='ml_dataset.json', epochs=100,
                                train_seed=None, out_dir='.'):
    """
    Train the N-model ensemble.

    The defaults reproduce the original behaviour exactly - 5 models, 100 epochs,
    ml_dataset.json, artifacts written to the repo root - so main.py's menu and
    any existing script calling this with no arguments get what they always got.
    The parameters exist because three things were previously impossible:

      dataset     cli.py accepted --dataset and this function ignored it,
                  hard-coding ml_dataset.json. A user pointing at another file
                  was silently trained on the wrong one.
      epochs      hard-coded 100, so there was no way to produce a deliberately
                  undertrained model to prove the promotion gate rejects it.
      train_seed  nothing seeded Keras, so two runs of the identical config
                  differed by an unmeasured amount. You cannot set a sensible
                  gate tolerance without knowing that spread, so this makes the
                  training run reproducible while leaving the evaluation split
                  (split.SPLIT_SEED, frozen at 42) completely untouched.
      out_dir     candidates now train into candidates/<name>/ and only reach
                  the repo root by being promoted, so a bad run cannot overwrite
                  the models the shipped app is using.

    Returns a dict describing the run so callers (the Prefect flow, the gate)
    can use the numbers instead of scraping stdout.
    """
    from mlops import split as split_mod

    print(f"--- Starting {num_models}-Model Ensemble Training ---")

    try:
        prepared = split_mod.prepare_dataset(dataset)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return None

    # Scaler fit on all rows, then split - the original order, preserved
    # deliberately. See split.py's docstring for why this leak stays.
    X_scaled, scaler = split_mod.scale_all(prepared)
    y_binary = prepared.y
    classes = prepared.classes

    sp = split_mod.fixed_split(prepared)
    X_train, X_test = X_scaled[sp.train_idx], X_scaled[sp.test_idx]
    y_train, y_test = y_binary[sp.train_idx], y_binary[sp.test_idx]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'ensemble_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, 'ensemble_binarizer.pkl'), 'wb') as f:
        pickle.dump(prepared_binarizer(prepared), f)
    split_mod.write_split_manifest(
        prepared, sp, os.path.join(out_dir, 'split_manifest.json'))

    if train_seed is not None:
        # Seeds python, numpy and tensorflow together. Note this makes the run
        # reproducible on the same machine and TF build; it is not a promise of
        # bit-identical results across platforms.
        tf.keras.utils.set_random_seed(train_seed)
        print(f"Training seed: {train_seed} (evaluation split seed stays {sp.seed})")

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

        model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                  validation_split=0.2, callbacks=[early_stopping], verbose=0)

        model_filename = os.path.join(out_dir, f'ensemble_model_{i+1}.keras')
        model.save(model_filename)
        trained_models.append(model)
        print(f"Model {i+1} saved as {model_filename}.")

    print("\n--- Generating Ensemble Predictions ---")
    all_predictions = [model.predict(X_test, verbose=0) for model in trained_models]
    averaged_probabilities = np.mean(all_predictions, axis=0)

    final_binary_predictions = (averaged_probabilities >= THRESHOLD).astype(int)

    print("\n" + "="*50)
    print(f"ENSEMBLE ({num_models} MODELS) CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, final_binary_predictions, target_names=classes, zero_division=0))

    return {
        'out_dir': out_dir,
        'num_models': num_models,
        'epochs': epochs,
        'train_seed': train_seed,
        'dataset': dataset,
        'dataset_sha256': prepared.dataset_sha,
        'split_hash': sp.split_hash,
        'n_train': int(len(sp.train_idx)),
        'n_holdout': int(len(sp.test_idx)),
        'classes': list(classes),
    }


def prepared_binarizer(prepared):
    """
    Rebuild the MultiLabelBinarizer that produced prepared.y.

    prepare_dataset() returns the fitted classes but not the object, and the app
    needs the object pickled. Refitting on the same tag lists is deterministic
    and yields identical classes_ - asserted here rather than assumed, because a
    mismatch would silently reorder the 66 outputs the C# app reads.
    """
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit(prepared.tag_lists)
    assert list(mlb.classes_) == list(prepared.classes), "binarizer classes drifted"
    return mlb


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
    if not sections:
        return ["Map is too short."]

    raw_features = classifier._aggregate_features_for_map(sections)
    if raw_features is None:
        return ["Feature extraction failed."]

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
