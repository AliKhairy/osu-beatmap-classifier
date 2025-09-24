# neural_model.py
"""
Defines the core machine learning model for the osu! Beatmap Classifier.

This module contains the `ImprovedBeatmapClassifier` class, which handles:
- Loading and saving the model.
- Advanced feature extraction from osu! hit objects.
- Splitting beatmaps into logical sections for analysis.
- Defining, training, and evaluating the neural network.
- Predicting tags for new beatmaps.
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from osu_parser import OsuFileParser

# --- Constants ---
CLASSIFIER_PATH = 'beatmap_classifier.pkl'

# Constants for feature indices to improve readability.
# This avoids using "magic numbers" when calculating derived scores.
IDX_STREAM_SCORE = 0
IDX_FINGER_CONTROL_SCORE = 1
IDX_TAPPING_DENSITY = 2
IDX_STREAM_PURITY_RATIO = 3
IDX_AVG_RHYTHM_INSTABILITY = 4
IDX_AVG_SPACING_INSTABILITY = 5
IDX_SLIDER_DISRUPTION_RATE = 6
IDX_FORMAL_STREAM_RATIO = 7
IDX_MEAN_DISTANCE = 12
IDX_STD_TIME_GAPS = 16
IDX_SLIDER_RATIO = 18
IDX_STD_ANGLES = 20
IDX_PERCENTILE_DIST_95 = 14


class ImprovedBeatmapClassifier:
    """
    Manages the entire lifecycle of the beatmap classification model.
    """

    def __init__(self):
        """Initializes the classifier's components."""
        self.model = None
        self.label_binarizer = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        self.is_trained = False

    def load_model(self):
        """
        Loads the entire classifier object (model, scaler, binarizer) from a pickle file.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        if not os.path.exists(CLASSIFIER_PATH):
            return False
        try:
            with open(CLASSIFIER_PATH, 'rb') as f:
                # Load the saved instance's attributes into the current instance
                loaded_classifier = pickle.load(f)
                self.__dict__.update(loaded_classifier.__dict__)
            self.is_trained = True
            print(f"Classifier loaded successfully from {CLASSIFIER_PATH}.")
            return True
        except (pickle.UnpicklingError, EOFError, ImportError, IndexError) as e:
            print(f"Error loading classifier from {CLASSIFIER_PATH}: {e}")
            return False

    def model_exists(self):
        """Checks if a saved model file already exists."""
        return os.path.exists(CLASSIFIER_PATH)

    def split_beatmap_into_sections(self, hit_objects):
        """
        Splits a beatmap's hit objects into playable sections based on breaks.

        A "break" is defined as a pause between hit objects longer than a
        certain threshold. This helps analyze sections like "kiai time" or
        intense choruses independently.

        Args:
            hit_objects (list): The list of raw hit objects from OsuFileParser.

        Returns:
            list: A list of sections, where each section is a list of hit objects.
        """
        if not hit_objects:
            return []

        sections, current_section = [], []
        times = [obj[2] for obj in hit_objects]
        # A 2000ms (2-second) pause is a reliable indicator of a gameplay break.
        break_threshold = 2000
        min_section_length = 15  # Ignore very short sections.

        for i, obj in enumerate(hit_objects):
            # Check for a long pause between the current and previous object.
            if i > 0 and (times[i] - times[i-1] > break_threshold):
                if len(current_section) >= min_section_length:
                    sections.append(current_section)
                current_section = []  # Start a new section
            current_section.append(obj)

        # Add the final section after the loop finishes.
        if len(current_section) >= min_section_length:
            sections.append(current_section)

        # If no breaks were found but the map is long enough, treat the whole map as one section.
        if not sections and len(hit_objects) >= min_section_length:
            return [hit_objects]

        return sections

    def extract_meaningful_features(self, hit_objects):
        """
        Converts a list of hit objects into a numerical feature vector.

        This is the core of the feature engineering process, calculating metrics
        related to rhythm, spacing, density, and patterns.

        Args:
            hit_objects (list): A list of hit objects for a single section.

        Returns:
            np.array: A NumPy array of 35 calculated features.
        """
        feature_count = 35
        # Require at least 5 objects to extract meaningful pattern data.
        if not hit_objects or len(hit_objects) < 5:
            return np.zeros(feature_count)

        times = np.array([obj[2] for obj in hit_objects])
        positions = np.array([(obj[0], obj[1]) for obj in hit_objects])
        is_slider = np.array([obj[4] is not None for obj in hit_objects])
        num_objects = len(hit_objects)

        total_duration = (times[-1] - times[0]) / 1000.0
        if total_duration == 0:
            return np.zeros(feature_count)

        # --- Foundational Metrics ---
        objects_per_sec = num_objects / total_duration
        time_gaps = np.diff(times)
        time_gaps[time_gaps == 0] = 1  # Avoid division by zero
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)

        features = []

        # --- Stream and Tapping Analysis ---
        # A time gap < 160ms corresponds to rhythms faster than ~187 BPM 1/4 notes, a common streaming threshold.
        dense_indices = np.where(time_gaps < 160)[0]
        tapping_density = len(dense_indices) / \
            num_objects if num_objects > 0 else 0

        rhythm_instabilities, spacing_instabilities, pure_stream_notes = [], [], 0
        if len(dense_indices) > 0:
            # Group consecutive dense notes into bursts/streams.
            groups = np.split(dense_indices, np.where(
                np.diff(dense_indices) != 1)[0] + 1)
            for group in groups:
                if len(group) < 2:
                    continue
                rhythm_std = np.std(time_gaps[group])
                spacing_std = np.std(distances[group])
                rhythm_instabilities.append(rhythm_std)
                spacing_instabilities.append(spacing_std)
                # "Pure" streams have very consistent rhythm (low std dev) and spacing.
                if rhythm_std < 7 and spacing_std < 40:
                    pure_stream_notes += len(group) + 1

        avg_rhythm_instability = np.mean(
            rhythm_instabilities) if rhythm_instabilities else 0
        avg_spacing_instability = np.mean(
            spacing_instabilities) if spacing_instabilities else 0
        # Measures how often sliders interrupt dense tapping patterns.
        slider_disruption_rate = np.sum(
            is_slider[1:-1] & (time_gaps[:-1] < 160) & (time_gaps[1:] < 160)) / num_objects
        # A composite score for patterns requiring precise finger control (complex rhythms, inconsistent spacing).
        finger_control_score = (avg_rhythm_instability * 1.2) + \
            (avg_spacing_instability * 0.3) + (slider_disruption_rate * 3.0)
        # The ratio of "pure" stream notes to all dense notes. High values indicate clean streams.
        stream_purity_ratio = pure_stream_notes / \
            len(dense_indices) if len(dense_indices) > 0 else 0
        # A score rewarding dense and pure stream patterns.
        stream_score = tapping_density * (stream_purity_ratio ** 2)

        # A stricter definition of streams (e.g., > 135ms or >222 BPM) and length.
        formal_stream_notes = 0
        current_len = 0
        for is_stream in (time_gaps < 135):
            if is_stream:
                current_len += 1
            else:
                if current_len >= 8:
                    formal_stream_notes += current_len + 1
                current_len = 0
        if current_len >= 8:
            formal_stream_notes += current_len + 1
        formal_stream_ratio = formal_stream_notes / num_objects

        features.extend([
            stream_score, finger_control_score, tapping_density, stream_purity_ratio,
            avg_rhythm_instability, avg_spacing_instability, slider_disruption_rate, formal_stream_ratio
        ])

        # --- General Pattern and Flow Analysis ---
        slider_ratio = np.sum(is_slider) / num_objects
        angles = np.array([])
        if num_objects > 2:
            v1 = positions[1:-1] - positions[:-2]
            v2 = positions[2:] - positions[1:-1]
            norm_prod = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
            valid_indices = norm_prod > 0
            if np.any(valid_indices):
                # Calculate the angle between three consecutive points to measure path curvature.
                cos_angles = np.clip(np.sum(
                    v1[valid_indices] * v2[valid_indices], axis=1) / norm_prod[valid_indices], -1.0, 1.0)
                angles = np.arccos(cos_angles)

        features.extend([
            num_objects, objects_per_sec,
            np.mean(distances) if distances.size > 0 else 0,
            np.std(distances) if distances.size > 0 else 0,
            # 95th percentile captures jump difficulty.
            np.percentile(distances, 95) if distances.size > 0 else 0,
            np.mean(time_gaps),
            np.std(time_gaps),
            slider_ratio,
            np.mean(angles) if angles.size > 0 else 0,  # Average angle change.
            # High std dev in angles indicates erratic aim patterns (tech).
            np.std(angles) if angles.size > 0 else 0
        ])

        # Pad with zeros if any features are missing to ensure a consistent vector length.
        while len(features) < feature_count:
            features.append(0)

        return np.array(features[:feature_count])

    def create_neural_network(self, input_dim, num_classes):
        """
        Defines and compiles the Keras neural network architecture.

        Args:
            input_dim (int): The number of input features.
            num_classes (int): The number of possible output tags.

        Returns:
            tf.keras.Model: The compiled Keras model.
        """
        model = tf.keras.Sequential([
            # Input layer with 128 neurons and ReLU activation.
            tf.keras.layers.Dense(128, activation='relu',
                                  input_shape=(input_dim,)),
            # Dropout layer to prevent overfitting.
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(32, activation='relu'),
            # Output layer with a 'sigmoid' activation for multi-label classification.
            tf.keras.layers.Dense(num_classes, activation='sigmoid')
        ])
        # Compile the model with Adam optimizer and binary cross-entropy for the loss function.
        model.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _aggregate_features_for_map(self, sections):
        """
        Aggregates features from all sections of a map into a single feature vector.
        This helper function is used by both train() and predict_tags() to ensure consistency.
        """
        section_features_list = [
            self.extract_meaningful_features(s) for s in sections]
        section_features_list = [
            f for f in section_features_list if f is not None]
        if not section_features_list:
            return None

        features_np = np.array(section_features_list)

        # --- Derived Tech Score ---
        # Combines features that indicate technical difficulty.
        section_tech_scores = (features_np[:, IDX_AVG_RHYTHM_INSTABILITY] * 1.5) + \
                              (features_np[:, IDX_AVG_SPACING_INSTABILITY] * 1.0) + \
                              (features_np[:, IDX_STD_ANGLES] * 0.8) + \
                              (features_np[:, IDX_SLIDER_RATIO] * 0.5) + \
                              (features_np[:, IDX_STD_TIME_GAPS] * 0.7)

        # --- Derived Flow Score ---
        # A more complex score rewarding smooth, stream-like patterns while penalizing erratic ones.
        flow_positive = (features_np[:, IDX_STREAM_SCORE] * 1.5) + \
                        (features_np[:, IDX_STREAM_PURITY_RATIO] * 1.0) + \
                        (features_np[:, IDX_FORMAL_STREAM_RATIO] * 1.0) + \
                        (1 / (1 + features_np[:, IDX_AVG_RHYTHM_INSTABILITY] * 0.5)) + \
                        (1 / (1 + features_np[:, IDX_AVG_SPACING_INSTABILITY] * 0.5)) + \
                        (1 / (1 + features_np[:, IDX_STD_ANGLES] * 0.5))
        flow_negative = (features_np[:, IDX_FINGER_CONTROL_SCORE] * 0.5) + \
                        (features_np[:, IDX_PERCENTILE_DIST_95] *
                         0.01)  # Penalize large jumps.
        section_flow_scores = flow_positive - flow_negative

        # --- Hybrid Flags ---
        # Binary flags indicating if a map contains sections with peak difficulty in certain skills.
        has_peak_stream_section = 1 if np.max(
            features_np[:, IDX_STREAM_SCORE]) > 0.15 else 0
        has_peak_fc_section = 1 if np.max(
            features_np[:, IDX_FINGER_CONTROL_SCORE]) > 1.5 else 0
        has_peak_jump_section = 1 if np.max(
            features_np[:, IDX_MEAN_DISTANCE]) > 200 else 0
        is_stream_jump_hybrid = 1 if has_peak_stream_section and has_peak_jump_section else 0

        # --- Final Aggregation ---
        # Combine max, mean, and std of all features across all sections, plus derived scores and flags.
        # This creates a comprehensive profile of the entire beatmap.
        aggregated_vector = np.concatenate([
            np.max(features_np, axis=0),
            np.mean(features_np, axis=0),
            np.std(features_np, axis=0),
            np.array([np.max(section_tech_scores), np.mean(
                section_tech_scores), np.std(section_tech_scores)]),
            np.array([np.max(section_flow_scores), np.mean(
                section_flow_scores), np.std(section_flow_scores)]),
            np.array([has_peak_stream_section, has_peak_fc_section,
                     has_peak_jump_section, is_stream_jump_hybrid])
        ])

        return aggregated_vector

    def train(self, dataset_filename='ml_dataset.json'):
        """
        Trains the neural network model on the provided dataset.

        This function handles data loading, preprocessing, feature aggregation,
        model training, and saving the final classifier.

        Args:
            dataset_filename (str): The path to the JSON dataset file.

        Returns:
            bool: True if training was successful, False otherwise.
        """
        try:
            with open(dataset_filename, 'r') as f:
                dataset = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(
                f"Error: Could not read dataset file '{dataset_filename}'. {e}")
            return False

        if len(dataset) < 10:
            print("Error: Dataset is too small. Need at least 10 beatmaps to train.")
            return False

        X_final, y_final = [], []
        print("Processing dataset and extracting features...")
        for sample in dataset:
            if sample.get('tags'):
                sections = self.split_beatmap_into_sections(
                    sample['hit_objects'])
                if not sections:
                    continue

                aggregated_vector = self._aggregate_features_for_map(sections)
                if aggregated_vector is None:
                    continue

                X_final.append(aggregated_vector)
                y_final.append(sample['tags'])

        if not X_final:
            print("Error: No valid data could be processed from the dataset.")
            return False

        # --- Data Preparation ---
        # Convert lists to NumPy arrays.
        X = np.array(X_final)
        # Scale features to have zero mean and unit variance. This is crucial for neural networks.
        X_scaled = self.scaler.fit_transform(X)
        # Convert the list of tag strings into a binary matrix format.
        y_binary = self.label_binarizer.fit_transform(y_final)
        # Split data into training and testing sets.
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_binary, test_size=0.2, random_state=42)

        # --- Model Training ---
        self.model = self.create_neural_network(
            X_train.shape[1], y_train.shape[1])
        print(f"Starting model training with {X_train.shape[1]} features...")

        # Use an EarlyStopping callback to prevent overfitting. Training stops if validation loss doesn't improve.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=20, restore_best_weights=True, monitor='val_loss')

        self.model.fit(X_train, y_train, epochs=120, batch_size=16, validation_data=(X_test, y_test), verbose=1,
                       callbacks=[early_stopping])

        _, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Model training complete. Test accuracy: {test_acc:.3f}")
        self.is_trained = True

        # --- Save the entire classifier object ---
        try:
            with open(CLASSIFIER_PATH, 'wb') as f:
                pickle.dump(self, f)
            print(f"Model saved successfully to {CLASSIFIER_PATH}.")
            return True
        except (IOError, pickle.PicklingError) as e:
            print(f"Error: Failed to save model. {e}")
            return False

    def predict_tags(self, osu_file_path, threshold=0.27):
        """
        Predicts tags for a single .osu file.

        Args:
            osu_file_path (str): The path to the .osu file to analyze.
            threshold (float): The probability threshold (0.0 to 1.0) required to assign a tag.

        Returns:
            list: A list of predicted tag strings.
        """
        if not self.is_trained:
            if not self.load_model():
                return ["Model not trained or loaded. Please train the model first."]

        parser = OsuFileParser(osu_file_path)
        parser.read_file()
        metadata = parser.get_metadata()
        print(
            f"\nPredicting for: {metadata.get('Artist')} - {metadata.get('Title')} [{metadata.get('Version')}]")

        # Use the same feature extraction pipeline as in training.
        sections = self.split_beatmap_into_sections(
            parser.extract_raw_hit_objects())
        if not sections:
            return ["Map is too short or has no playable sections."]

        aggregated_vector = self._aggregate_features_for_map(sections)
        if aggregated_vector is None:
            return ["Feature extraction failed for this map."]

        # Reshape for a single prediction and scale using the already-fitted scaler.
        aggregated_vector = aggregated_vector.reshape(1, -1)
        X_scaled = self.scaler.transform(aggregated_vector)

        # Get prediction probabilities from the model.
        probabilities = self.model.predict(X_scaled, verbose=0)[0]

        predicted_tags = []
        print("\nPrediction Probabilities:")
        # Display each tag and its corresponding probability.
        for i, tag in enumerate(self.label_binarizer.classes_):
            prob = probabilities[i]
            is_predicted = prob >= threshold
            print(
                f"  [{'x' if is_predicted else ' '}] {tag:<20} | Probability: {prob:.3f}")
            if is_predicted:
                predicted_tags.append(tag)

        return predicted_tags if predicted_tags else ["No tags above threshold."]

    def test_multiple_maps(self, songs_folder="songs", threshold=0.27, max_maps=10):
        """
        A utility function to run predictions on multiple maps in a folder.
        """
        if not self.is_trained:
            if not self.load_model():
                print(
                    "Error: Model not trained or loaded. Please train the model first.")
                return

        osu_files = [f for f in os.listdir(songs_folder) if f.endswith('.osu')]
        if not osu_files:
            print(
                f"Error: No .osu files found in the '{songs_folder}' directory.")
            return

        num_to_test = min(len(osu_files), max_maps)
        print(f"\n--- Testing model on {num_to_test} maps ---")

        for i, filename in enumerate(osu_files[:num_to_test]):
            map_path = os.path.join(songs_folder, filename)
            print(f"\n--- Test {i+1}/{num_to_test} ---")
            predicted_tags = self.predict_tags(map_path, threshold)
            print(f"Final prediction: {predicted_tags}")
