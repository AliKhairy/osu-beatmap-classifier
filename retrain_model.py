# retrain_model.py
"""
This script provides a mechanism for retraining the neural network with corrected data.

It reads a 'corrections.json' file, which is expected to contain beatmaps and their
manually corrected tags. The script then updates the main 'ml_dataset.json' with
these corrections—either by modifying existing entries or adding new ones. After
updating the dataset, it triggers a full retraining of the model using the
ImprovedBeatmapClassifier.

This creates a valuable feedback loop for continuously improving the model's accuracy.
"""

import os
import sys
import json
from neural_model import ImprovedBeatmapClassifier
from osu_parser import OsuFileParser


def retrain_with_corrections(project_path='.'):
    """
    Loads corrections, updates the main dataset, and initiates model retraining.

    Args:
        project_path (str): The root directory of the project where the dataset
                            and corrections files are located. Defaults to the
                            current directory.

    Returns:
        str: A message summarizing the result of the operation.
    """
    # Construct full paths to the required files for robustness.
    dataset_path = os.path.join(project_path, 'ml_dataset.json')
    corrections_path = os.path.join(project_path, 'corrections.json')

    # --- Step 1: Validate that necessary files exist ---
    if not os.path.exists(corrections_path):
        return "No 'corrections.json' file found. Nothing to do."

    if not os.path.exists(dataset_path):
        return "Main 'ml_dataset.json' not found. Please build it first."

    # --- Step 2: Load the existing dataset and the new corrections ---
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            main_dataset = json.load(f)

        with open(corrections_path, 'r', encoding='utf-8') as f:
            corrections = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        return f"Error reading data files: {e}"

    print(f"Loaded {len(main_dataset)} existing dataset entries.")
    print(f"Found {len(corrections)} corrections to apply.")

    # Convert the dataset list to a dictionary for efficient lookups using beatmap_id.
    dataset_dict = {
        item['beatmap_id']: item for item in main_dataset if 'beatmap_id' in item
    }

    updated_count = 0
    added_count = 0

    # --- Step 3: Process each correction ---
    for correction in corrections:
        try:
            map_path = correction['map_path']
            corrected_tags = correction['corrected_tags']

            if not os.path.exists(map_path):
                print(
                    f"Warning: Map file not found for correction, skipping: {map_path}")
                continue

            # Parse the .osu file to get its unique BeatmapID.
            parser = OsuFileParser(map_path)
            parser.read_file()
            beatmap_id = parser.get_beatmap_id()

            if not beatmap_id:
                print(
                    f"Warning: Could not find BeatmapID in {map_path}, skipping correction.")
                continue

            # Check if this beatmap is already in our dataset.
            if beatmap_id in dataset_dict:
                # If it exists, update its tags with the corrected ones.
                print(f"Updating tags for BeatmapID: {beatmap_id}")
                dataset_dict[beatmap_id]['tags'] = corrected_tags
                updated_count += 1
            else:
                # If it's a new map, parse it and add it as a new entry.
                print(f"Adding new entry for BeatmapID: {beatmap_id}")
                hit_objects = parser.extract_raw_hit_objects()
                metadata = parser.get_metadata()
                new_entry = {
                    'beatmap_id': beatmap_id,
                    'title': metadata.get('Title', 'Unknown Title'),
                    'hit_objects': hit_objects,
                    'tags': corrected_tags
                }
                dataset_dict[beatmap_id] = new_entry
                added_count += 1
        except (KeyError, Exception) as e:
            print(
                f"Error processing a correction entry: {correction}. Error: {e}")

    # --- Step 4: Save the updated dataset ---
    # Convert the dictionary back to a list before saving.
    updated_dataset = list(dataset_dict.values())
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(updated_dataset, f, indent=2)

    print("\nDataset update complete.")
    print(f" - {updated_count} entries modified.")
    print(f" - {added_count} new entries added.")
    print("--- Starting model retraining... ---")

    # --- Step 5: Retrain the model with the improved dataset ---
    classifier = ImprovedBeatmapClassifier()
    # The train method handles loading, processing, training, and saving the new model.
    success = classifier.train(dataset_path)

    if success:
        # Optional: You could uncomment the next line to automatically delete
        # the corrections file after it has been successfully applied.
        # os.remove(corrections_path)
        return f"\nModel retrained successfully on {len(updated_dataset)} maps!"
    else:
        return "\nModel retraining failed."


if __name__ == "__main__":
    # Allows running the script with an optional path argument, e.g., `python retrain_model.py /path/to/project`
    project_path_arg = sys.argv[1] if len(sys.argv) > 1 else '.'
    result_message = retrain_with_corrections(project_path_arg)
    print(result_message)
