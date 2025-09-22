# dataset_builder.py
"""
This script constructs the machine learning dataset by repeatedly calling the
complete_pipeline module. It is designed to be run to collect a large number
of processed beatmap samples and save them into a single JSON file.

It includes functionality to load an existing dataset and append new data,
making it easy to resume or expand the data collection process.
"""

import time
import os
import json
from complete_pipeline import get_oauth_token, process_one_api_map


def build_dataset(num_maps=10, start_offset=0):
    """
    Builds a dataset by processing a specified number of maps from the API.

    This function loops 'num_maps' times, calling the processing pipeline for
    each map. It starts from a given 'start_offset' in the Echo API database.
    It also includes basic error handling to skip problematic maps and to stop
    if too many consecutive errors occur.

    Args:
        num_maps (int): The total number of maps to attempt to process.
        start_offset (int): The starting index (offset) in the Echo API
                            database from which to begin fetching maps.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              successfully processed beatmap sample. Returns an empty list
              if the API token cannot be obtained.
    """
    dataset = []
    failed_count = 0

    # First, obtain the necessary API token to download .osu files.
    token = get_oauth_token()
    if not token:
        print("Failed to get OAuth token. Cannot build dataset.")
        return []

    # Loop for the desired number of maps.
    for i in range(num_maps):
        # Calculate the current offset to fetch the next unique map from the API.
        current_offset = start_offset + i
        print(
            f"\n--- Processing map {i + 1}/{num_maps} (API offset: {current_offset}) ---"
        )

        try:
            # Execute the full pipeline for a single map.
            sample = process_one_api_map(token, offset=current_offset)

            # A sample is only valid if it was processed successfully and has tags.
            if sample and sample.get('tags'):
                dataset.append(sample)
                print("Successfully added beatmap to dataset.")
            else:
                print("Skipped map (processing failed or no relevant tags found).")

        except Exception as e:
            # Catch any unexpected errors during the pipeline process.
            print(
                f"An unexpected error occurred while processing map {i + 1}: {e}")
            failed_count += 1

            # If too many consecutive errors occur, stop the process.
            if failed_count > 10:
                print("Stopping due to an excessive number of consecutive failures.")
                break

            # Wait briefly after an error to avoid spamming the API.
            time.sleep(2)

    return dataset


# This block of code runs when the script is executed directly.
if __name__ == "__main__":
    DATASET_FILENAME = 'ml_dataset.json'
    MAPS_TO_DOWNLOAD = 1000
    DOWNLOAD_START_OFFSET = 1000

    print("--- Dataset Builder ---")

    # Step 1: Load any existing dataset to avoid overwriting previous work.
    existing_data = []
    if os.path.exists(DATASET_FILENAME):
        try:
            with open(DATASET_FILENAME, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(
                f"Loaded {len(existing_data)} existing maps from {DATASET_FILENAME}.")
        except (json.JSONDecodeError, IOError) as e:
            print(
                f"Could not load existing dataset. Starting fresh. Error: {e}")

    # Step 2: Build the new dataset, starting from the specified offset.
    # For example, if you already have 1000 maps, you can set start_offset=1000
    # to continue where you left off.
    print(
        f"Attempting to download {MAPS_TO_DOWNLOAD} new maps, starting from offset {DOWNLOAD_START_OFFSET}.")
    new_data = build_dataset(
        MAPS_TO_DOWNLOAD, start_offset=DOWNLOAD_START_OFFSET)

    # Step 3: Combine the newly downloaded data with the existing data.
    combined_dataset = existing_data + new_data

    # Step 4: Save the complete dataset back to the JSON file.
    # The `indent=2` argument makes the JSON file human-readable.
    try:
        with open(DATASET_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(combined_dataset, f, indent=2)
        print(f"\nDataset construction complete.")
        print(f"Total maps in dataset: {len(combined_dataset)}")
        print(f"Saved all data to {DATASET_FILENAME}.")
    except IOError as e:
        print(f"Failed to save the dataset. Error: {e}")
