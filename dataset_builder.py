# dataset_builder.py (Final Version with Offset)
"""
This script constructs a dataset by fetching the entire list of beatmaps
from the echosu.com API and processing a specific slice of that list.
"""
import time
import os
import json
from dotenv import load_dotenv

from complete_pipeline import get_oauth_token, get_beatmap_file
from echosu_api import EchoOsuAPI
from osu_parser import OsuFileParser
from tag_scraper import filter_tags, tag_counts

load_dotenv()
ECHO_API_TOKEN = os.getenv("ECHO_API_TOKEN")

def build_full_dataset(max_maps=500, offset=0):
    if not ECHO_API_TOKEN:
        print("Error: ECHO_API_TOKEN not found.")
        return []

    osu_api_token = get_oauth_token()
    if not osu_api_token:
        print("Failed to get osu! API token.")
        return []

    echo_api = EchoOsuAPI(ECHO_API_TOKEN)
    full_dataset = []

    # 1. Make one single request to get the entire list of maps.
    all_maps = echo_api.discover_all_beatmaps()

    if not all_maps:
        print("Could not discover any maps. Aborting.")
        return []

    # We select a chunk of the list starting from the offset.
    end_index = offset + max_maps
    maps_to_process = all_maps[offset:end_index]

    total_to_process = len(maps_to_process)
    print(
        f"Full list contains {len(all_maps)} maps. Processing slice from index {offset} to {end_index}.")

    # 2. Loop through the selected slice of maps.
    for i, map_data in enumerate(maps_to_process):
        beatmap_id = map_data.get('beatmap_id')
        title = map_data.get('title', 'Unknown Title')

        if not beatmap_id:
            continue

        print(
            f"\n--- Processing map {i + 1}/{total_to_process}: {title} (ID: {beatmap_id}) ---")

        original_tags_list = map_data.get('tags', [])
        if not original_tags_list:
            print("  Skipping: No tags found for this map.")
            continue

        unique_tag_names = {tag['name']
                            for tag in original_tags_list if 'name' in tag}
        tags_data = [{'tag': name} for name in unique_tag_names]

        osu_file_path = get_beatmap_file(beatmap_id, osu_api_token)
        if not osu_file_path:
            print("  Skipping: Failed to download .osu file.")
            continue

        parser = OsuFileParser(osu_file_path)
        parser.read_file()
        hit_objects = parser.extract_raw_hit_objects()
        filtered_tags = filter_tags(tags_data, tag_counts)

        if filtered_tags:
            sample = {
                'beatmap_id': beatmap_id,
                'title': title,
                'hit_objects': hit_objects,
                'tags': filtered_tags
            }
            full_dataset.append(sample)
            print(
                f"  Successfully added to dataset. Total maps in this run: {len(full_dataset)}")
        else:
            print("  Skipping: No relevant tags after filtering.")

        time.sleep(0.5)

    return full_dataset


if __name__ == "__main__":
    DATASET_FILENAME = 'ml_dataset.json'
    # How many maps to process in this run.
    MAPS_TO_PROCESS_IN_RUN = 500
    # Where to start in the full list. Change this to resume.
    # 0 = start from the beginning
    # 500 = skip the first 500 and start there
    START_OFFSET = 0

    print("--- Full Dataset Builder ---")

    new_data = build_full_dataset(
        max_maps=MAPS_TO_PROCESS_IN_RUN, offset=START_OFFSET)

    # This will overwrite the dataset file. You can add logic to append if you prefer.
    with open(DATASET_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)

    print(
        f"\nDataset construction complete. Saved {len(new_data)} maps to {DATASET_FILENAME}.")
