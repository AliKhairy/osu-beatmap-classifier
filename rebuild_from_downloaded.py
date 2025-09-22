# rebuild_from_downloaded.py
"""
This script rebuilds the 'ml_dataset.json' from a collection of local .osu files.

Its primary purpose is to create a fresh, high-quality dataset by:
1.  Iterating through a folder of downloaded .osu files.
2.  Parsing each file to extract its BeatmapID and hit objects.
3.  Filtering out maps that are excessively long (e.g., marathons).
4.  Using the BeatmapID to fetch the latest community tags from the Echo API.
5.  Filtering these tags to keep only the most relevant and popular ones.
6.  Saving the combined, cleaned data into a new 'ml_dataset.json' file.

This is useful for ensuring the dataset is based on the most current tags and for
curating the data from a specific collection of beatmaps.
"""

import os
import json
import time
from dotenv import load_dotenv
from osu_parser import OsuFileParser
from echosu_api import EchoOsuAPI
from tag_scraper import tag_counts, filter_tags

# --- Configuration ---
# Load environment variables from the .env file (e.g., for API keys).
load_dotenv()
ECHO_TOKEN = os.getenv("ECHO_API_TOKEN")

# The folder containing your collection of .osu files to be processed.
LOCAL_OSU_FOLDER = "downloads"
OUTPUT_DATASET_FILE = "ml_dataset.json"

# A filter to exclude maps longer than this duration in minutes.
# This helps prevent long marathon maps from disproportionately affecting the dataset.
MAX_MAP_MINUTES = 6.1


def rebuild():
    """Main function to orchestrate the dataset rebuilding process."""
    if not ECHO_TOKEN:
        print("Error: ECHO_API_TOKEN not found. Please create a .env file.")
        return

    print(
        f"Starting dataset rebuild from local files in '{LOCAL_OSU_FOLDER}'...")

    if not os.path.exists(LOCAL_OSU_FOLDER):
        print(
            f"Error: Folder '{LOCAL_OSU_FOLDER}' not found. Please create it and add .osu files.")
        return

    echo_api = EchoOsuAPI(ECHO_TOKEN)
    new_dataset = []
    osu_files = [f for f in os.listdir(LOCAL_OSU_FOLDER) if f.endswith('.osu')]
    total_files = len(osu_files)

    if total_files == 0:
        print(f"No .osu files found in '{LOCAL_OSU_FOLDER}'. Aborting.")
        return

    print(f"Found {total_files} local .osu files to process.")

    for i, filename in enumerate(osu_files):
        filepath = os.path.join(LOCAL_OSU_FOLDER, filename)
        print(f"\n--- Processing {i + 1}/{total_files}: {filename} ---")

        try:
            # Step 1: Parse the local .osu file.
            parser = OsuFileParser(filepath)
            parser.read_file()

            hit_objects = parser.extract_raw_hit_objects()
            if len(hit_objects) < 10:
                print("  Skipping: Map has too few hit objects.")
                continue

            # Step 2: Check and filter by map duration.
            start_time = hit_objects[0][2]  # Time of the first object in ms
            end_time = hit_objects[-1][2]   # Time of the last object in ms
            duration_minutes = (end_time - start_time) / 60000.0

            if duration_minutes > MAX_MAP_MINUTES:
                print(
                    f"  Skipping: Map is too long ({duration_minutes:.1f} min).")
                continue

            # Step 3: Get BeatmapID and fetch tags from the Echo API.
            beatmap_id = parser.get_beatmap_id()
            if not beatmap_id:
                print("  Skipping: Could not find BeatmapID in the file.")
                continue

            print(f"  Found BeatmapID: {beatmap_id}. Fetching tags...")
            # Add a small delay to be respectful to the API.
            time.sleep(0.5)
            api_tags_data = echo_api.get_beatmap_tags(beatmap_id)
            if not api_tags_data:
                print("  Skipping: No tags found from API.")
                continue

            # Step 4: Filter the tags to keep only relevant ones.
            filtered_tags = filter_tags(api_tags_data, tag_counts)
            print(f"  Filtered tags: {filtered_tags}")
            if not filtered_tags:
                print("  Skipping: No relevant tags remained after filtering.")
                continue

            # Step 5: (Optional) Clean up redundant or conflicting tags.
            # For example, 'stream section' is more specific than 'streams', so we prefer it.
            cleaned_tags = list(filtered_tags)
            if 'stream section' in cleaned_tags:
                if 'streams' in cleaned_tags:
                    cleaned_tags.remove('streams')
                if 'spaced streams' in cleaned_tags:
                    cleaned_tags.remove('spaced streams')

            # Step 6: Assemble the final data sample and add it to the new dataset.
            metadata = parser.get_metadata()
            title = metadata.get('Title', filename)

            sample = {
                'beatmap_id': beatmap_id,
                'title': title,
                'hit_objects': hit_objects,
                'tags': cleaned_tags
            }
            new_dataset.append(sample)
            print(f"  Successfully added '{title}' to the new dataset.")

        except Exception as e:
            print(f"  An error occurred while processing {filename}: {e}")

    # Step 7: Save the completed new dataset to the output file.
    print(
        f"\nRebuild complete. Created a dataset with {len(new_dataset)} maps.")
    with open(OUTPUT_DATASET_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, indent=2)

    print(f"Successfully saved new dataset to '{OUTPUT_DATASET_FILE}'.")


if __name__ == "__main__":
    rebuild()
