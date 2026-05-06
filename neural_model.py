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
        Converts a list of hit objects into a numerical feature vector,
        incorporating Global Snap Variance to detect complex rhythms.
        """
        
        if not hit_objects or len(hit_objects) < 5:
            return np.zeros(feature_count)

        times = np.array([obj[2] for obj in hit_objects])
        positions = np.array([(obj[0], obj[1]) for obj in hit_objects])
        is_slider = np.array([obj[4] is not None for obj in hit_objects])
        num_objects = len(hit_objects)

        total_duration = (times[-1] - times[0]) / 1000.0
        if total_duration == 0:
            return np.zeros(feature_count)

        objects_per_sec = num_objects / total_duration
        time_gaps = np.diff(times)
        time_gaps[time_gaps == 0] = 1 
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)

        features = []

        # --- ABSOLUTE SEQUENCE TRACKING (The Hybrid Fix) ---
        sequence_lengths = []
        current_len = 0
        
        # Distance < 120 ensures we don't count high-BPM cross-screen jumps as "streams".
        for gap, dist in zip(time_gaps, distances):
            if gap < 165 and dist < 120:
                current_len += 1
            else:
                if current_len > 0:
                    sequence_lengths.append(current_len + 1)
                current_len = 0
        if current_len > 0:
            sequence_lengths.append(current_len + 1)

        burst_count = sum(1 for l in sequence_lengths if 3 <= l <= 7)
        stream_count = sum(1 for l in sequence_lengths if l >= 8)
        # --- VARIABLE STREAMS (Spacing Variance) ---
        # Find the maximum spacing instability within any single stream
        rhythm_instabilities, spacing_instabilities = [], []
        dense_indices = np.where(time_gaps < 165)[0]
        max_stream_spacing_variance = 0
        if len(dense_indices) > 0:
            groups = np.split(dense_indices, np.where(np.diff(dense_indices) != 1)[0] + 1)
            for group in groups:
                if len(group) >= 8: # If it's a stream
                    stream_spacing_std = np.std(distances[group])
                    max_stream_spacing_variance = max(max_stream_spacing_variance, stream_spacing_std)

        # --- BUZZ SLIDERS ---
        # sliders with > 3 repeats (slides > 4 means it goes back and forth multiple times)
        buzz_slider_count = 0
        for obj in hit_objects:
            if obj[4] is not None: # is slider
                slides = obj[6]
                length = obj[7]
                # High repeats + short pixel length = buzz slider
                if slides >= 4 and length < 100: 
                    buzz_slider_count += 1
        max_continuous_stream = max(sequence_lengths) if sequence_lengths else 0
        total_stream_notes = sum(l for l in sequence_lengths if l >= 8)

        # --- GLOBAL SNAP VARIANCE (The Finger Control/Tech Fix) ---
        # Detects when a mapper shifts between 1/2, 1/3, 1/4, and 1/6 snaps.
        active_gaps = time_gaps[time_gaps < 750] # Focus on active gameplay, ignore breaks
        if len(active_gaps) > 1:
            # Calculate the absolute difference between consecutive gaps
            gap_diffs = np.abs(np.diff(active_gaps))
            # If the gap changes by >15ms, it's a deliberate rhythm/snap change
            rhythm_change_ratio = np.sum(gap_diffs > 15) / len(active_gaps)
            global_rhythm_variance = np.std(active_gaps)
        else:
            rhythm_change_ratio = 0
            global_rhythm_variance = 0

        # --- Local Instability Metrics ---
        rhythm_instabilities, spacing_instabilities = [], []
        dense_indices = np.where(time_gaps < 165)[0]
        
        if len(dense_indices) > 0:
            groups = np.split(dense_indices, np.where(np.diff(dense_indices) != 1)[0] + 1)
            for group in groups:
                if len(group) < 2: continue
                rhythm_instabilities.append(np.std(time_gaps[group]))
                spacing_instabilities.append(np.std(distances[group]))

        avg_rhythm_instability = np.mean(rhythm_instabilities) if rhythm_instabilities else 0
        avg_spacing_instability = np.mean(spacing_instabilities) if spacing_instabilities else 0
        slider_disruption_rate = np.sum(is_slider[1:-1] & (time_gaps[:-1] < 160) & (time_gaps[1:] < 160)) / num_objects
        finger_control_score = (avg_rhythm_instability * 1.2) + (avg_spacing_instability * 0.3) + (slider_disruption_rate * 3.0)

        # --- MICRO-PATTERNS & GEOMETRY ---
        slider_ratio = np.sum(is_slider) / num_objects
        
        # Angle Buckets (in radians. Pi = 3.14 = 180 degrees)
        sharp_angles, square_angles, wide_angles, linear_angles = 0, 0, 0, 0
        angles = np.array([])
        
        if num_objects > 2:
            v1 = positions[1:-1] - positions[:-2]
            v2 = positions[2:] - positions[1:-1]
            norm_prod = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
            valid_indices = norm_prod > 0
            
            if np.any(valid_indices):
                cos_angles = np.clip(np.sum(v1[valid_indices] * v2[valid_indices], axis=1) / norm_prod[valid_indices], -1.0, 1.0)
                angles = np.arccos(cos_angles)
                
                # Categorize the angles
                sharp_angles = np.sum(angles < 1.04)   # < 60 degrees (Snap Aim / Awkward)
                square_angles = np.sum((angles > 1.3) & (angles < 1.8)) # ~90 degrees (Square Jumps)
                wide_angles = np.sum((angles > 2.09) & (angles < 2.6))  # > 120 degrees (Flow Aim)
                linear_angles = np.sum(angles > 2.7)   # ~180 degrees (Linear Aim / 1-2 Jumps)

        # Vertical Jumps: Large Y movement, tiny X movement
        dx = np.abs(np.diff(positions[:, 0]))
        dy = np.abs(np.diff(positions[:, 1]))
        vertical_jumps = np.sum((dy > 120) & (dx < 40))

        # Perfect Overlaps: Object is placed exactly where an object was 2 steps ago
        perfect_overlaps = 0
        if num_objects > 2:
            dist_2_steps_back = np.linalg.norm(positions[2:] - positions[:-2], axis=1)
            perfect_overlaps = np.sum(dist_2_steps_back < 10) # Less than 10 pixels away

        # --- TRUE LINEAR PATTERN DETECTION (Collinearity) ---
        # Look at chunks of 4 consecutive hit objects.
        # If the points form a long, thin bounding box, they are a linear pattern.
        true_linear_sequences = 0
        if num_objects >= 4:
            for i in range(num_objects - 3):
                chunk = positions[i:i+4]
                
                # Find the bounding box width and height
                x_spread = np.max(chunk[:, 0]) - np.min(chunk[:, 0])
                y_spread = np.max(chunk[:, 1]) - np.min(chunk[:, 1])
                
                # We need to find the principal axis (the length of the line)
                # and the orthogonal spread (how "fat" the line is).
                # A simple approximation: max spread vs min spread.
                # However, a perfect diagonal has equal X and Y spread!
                
                # Better approach: check the distance from points to the line connecting start and end.
                start_pt = chunk[0]
                end_pt = chunk[-1]
                line_vec = end_pt - start_pt
                line_len = np.linalg.norm(line_vec)
                
                if line_len > 50: # The sequence must actually cover some distance
                    # Normalize the line vector
                    line_dir = line_vec / line_len
                    # Normal vector (perpendicular to the line)
                    normal_vec = np.array([-line_dir[1], line_dir[0]])
                    
                    # Calculate how far the middle two points deviate from the straight line
                    dev1 = np.abs(np.dot(chunk[1] - start_pt, normal_vec))
                    dev2 = np.abs(np.dot(chunk[2] - start_pt, normal_vec))
                    
                    # If both middle points are very close to the line (less than 15 pixels off), it's linear.
                    if dev1 < 15 and dev2 < 15:
                        true_linear_sequences += 1

        # --- THE FINAL FEATURE VECTOR (Exactly 29 Features) ---
        # We define this as a direct list to strictly control the indices for aggregation.
        features = [
            # 0-3: Stream Counters
            burst_count, stream_count, max_continuous_stream, total_stream_notes,
            # 4-5: Global Rhythm (Finger Control)
            rhythm_change_ratio, global_rhythm_variance,
            # 6-7: Slider & Stream Spacing Variance
            max_stream_spacing_variance, buzz_slider_count,
            # 8-11: Local Instability (Tech/Reading)
            finger_control_score, avg_rhythm_instability, avg_spacing_instability, slider_disruption_rate,
            # 12-16: Distance & Jump Geography
            num_objects, objects_per_sec, 
            np.mean(distances) if distances.size > 0 else 0,
            np.std(distances) if distances.size > 0 else 0,
            np.percentile(distances, 95) if distances.size > 0 else 0, # <-- This is now Index 16
            # 17-19: Time Gaps & Base Sliders
            np.mean(time_gaps), np.std(time_gaps), slider_ratio,
            # 20-21: Base Angles
            np.mean(angles) if angles.size > 0 else 0,
            np.std(angles) if angles.size > 0 else 0,
            # 22-28: Micro-patterns & Geometry
            sharp_angles / num_objects if num_objects > 0 else 0,
            square_angles / num_objects if num_objects > 0 else 0,
            wide_angles / num_objects if num_objects > 0 else 0,
            linear_angles / num_objects if num_objects > 0 else 0,
            vertical_jumps / num_objects if num_objects > 0 else 0,
            perfect_overlaps / num_objects if num_objects > 0 else 0,
            true_linear_sequences / num_objects if num_objects > 0 else 0
        ]

        return np.array(features)

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

        # --- Hybrid Flags ---
        # IDX 2 is 'max_continuous_stream'
        has_peak_stream_section = 1 if np.max(features_np[:, 2]) >= 12 else 0
        
        # IDX 16 is 'percentile_dist_95' (large jumps)
        has_peak_jump_section = 1 if np.max(features_np[:, 16]) > 180 else 0
        
        # Explicit Hybrid Override
        is_stream_jump_hybrid = 1 if has_peak_stream_section and has_peak_jump_section else 0

        # --- Final Aggregation ---
        # Combine max, mean, and std of all base features across all sections.
        # We removed manual heuristic scores to allow the Keras Dense layers to 
        # map the non-linear relationships natively.
        aggregated_vector = np.concatenate([
            np.max(features_np, axis=0),
            np.mean(features_np, axis=0),
            np.std(features_np, axis=0),
            np.array([has_peak_stream_section, has_peak_jump_section, is_stream_jump_hybrid])
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
        Predicts tags for a single .osu file, with post-processing 
        to correct neural network bias on hybrid maps.
        """
        if not self.is_trained:
            if not self.load_model():
                return ["Model not trained or loaded. Please train the model first."]

        parser = OsuFileParser(osu_file_path)
        parser.read_file()
        metadata = parser.get_metadata()
        print(f"\nPredicting for: {metadata.get('Artist')} - {metadata.get('Title')} [{metadata.get('Version')}]")

        sections = self.split_beatmap_into_sections(parser.extract_raw_hit_objects())
        if not sections:
            return ["Map is too short or has no playable sections."]

        aggregated_vector = self._aggregate_features_for_map(sections)
        if aggregated_vector is None:
            return ["Feature extraction failed for this map."]

        # Keep a copy of the unscaled features to use for Expert Rules
        raw_features = aggregated_vector.copy()

        aggregated_vector = aggregated_vector.reshape(1, -1)
        X_scaled = self.scaler.transform(aggregated_vector)
        probabilities = self.model.predict(X_scaled, verbose=0)[0]

        predicted_tags = []
        print("\nPrediction Probabilities:")
        for i, tag in enumerate(self.label_binarizer.classes_):
            prob = probabilities[i]
            is_predicted = prob >= threshold
            print(f"  [{'x' if is_predicted else ' '}] {tag:<20} | Probability: {prob:.3f}")
            if is_predicted:
                predicted_tags.append(tag)

        # --- EXPERT SYSTEM POST-PROCESSING (Hybrid Bias Correction) ---
        max_stream_length = raw_features[2] 
        
        # We only force the stream tag if the map is NOT an alt map.
        # Find the probability of 'alternating' to act as a safety check.
        alt_prob = 0.0
        if 'alternating' in self.label_binarizer.classes_:
            alt_index = list(self.label_binarizer.classes_).index('alternating')
            alt_prob = probabilities[alt_index]

        # If it's a 15+ note sequence AND it's not strongly an alt map
        if max_stream_length >= 15 and alt_prob < 0.35:
            if 'streams' not in predicted_tags and 'streams' in self.label_binarizer.classes_:
                predicted_tags.append('streams')
                print(f"\n[!] OVERRIDE: Network bias corrected. Map contains a {int(max_stream_length)}-note stream. Forcing 'streams' tag.")

        # Sort alphabetically for clean output
        predicted_tags.sort()
        
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
