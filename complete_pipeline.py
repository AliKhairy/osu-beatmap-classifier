# complete_pipeline.py
"""
This script orchestrates the complete data collection pipeline for the osu! beatmap classifier.
It fetches beatmap metadata from the Echo API, downloads the corresponding .osu file using the
official osu! API v2, parses the file to extract hit object data, and filters the tags to create
a single, clean training sample.
"""

import os
import requests
from echosu_api import EchoOsuAPI
from osu_parser import OsuFileParser
from tag_scraper import tag_counts, filter_tags
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

# Get tokens from the environment
ECHO_TOKEN = os.getenv("ECHO_API_TOKEN")
OSU_CLIENT_ID = os.getenv("OSU_CLIENT_ID")
OSU_CLIENT_SECRET = os.getenv("OSU_CLIENT_SECRET")


def get_oauth_token():
    """
    Authenticates with the osu! API v2 to obtain an OAuth access token.

    This token is required for making authorized requests to endpoints like
    downloading beatmap files.

    Returns:
        str: The access token if authentication is successful.
        None: If authentication fails.
    """
    # Define the data payload required for the client credentials grant type.
    data = {
        'client_id': OSU_CLIENT_ID,
        'client_secret': OSU_CLIENT_SECRET,
        'grant_type': 'client_credentials',
        # 'public' scope is sufficient for read-only actions.
        'scope': 'public'
    }

    # The endpoint for obtaining an OAuth token.
    token_url = 'https://osu.ppy.sh/oauth/token'

    try:
        # Make a POST request to the osu! API to get the token.
        response = requests.post(token_url, data=data)
        # Raise an exception for bad status codes (4xx or 5xx).
        response.raise_for_status()

        token_data = response.json()
        print("Successfully obtained osu! API OAuth token.")
        return token_data.get('access_token')

    except requests.exceptions.RequestException as e:
        print(f"Error obtaining OAuth token: {e}")
        return None


def get_beatmap_file(beatmap_id, token, save_folder="downloads"):
    """
    Downloads the .osu file for a given beatmap ID and saves it locally.

    Args:
        beatmap_id (str or int): The ID of the beatmap to download.
        token (str): The OAuth access token for the osu! API.
        save_folder (str): The local folder where the .osu file will be saved.

    Returns:
        str: The full file path of the saved .osu file if successful.
        None: If the download fails.
    """
    # The API endpoint for downloading a specific .osu file.
    download_url = f'https://osu.ppy.sh/osu/{beatmap_id}'
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/octet-stream'  # Specify the desired content type
    }

    # Ensure the target directory for saving the file exists.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Created directory: {save_folder}")

    try:
        response = requests.get(download_url, headers=headers)
        response.raise_for_status()

        # Define a standard filename for the downloaded map.
        filename = f"downloaded_{beatmap_id}.osu"
        filepath = os.path.join(save_folder, filename)

        # Write the file content to the local disk.
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"Successfully saved beatmap to: {filepath}")
        return filepath

    except requests.exceptions.RequestException as e:
        print(f"Failed to download .osu file for beatmap ID {beatmap_id}: {e}")
        return None


def process_one_api_map(token, offset=0):
    """
    Executes the full pipeline for a single beatmap fetched from the Echo API.

    This involves fetching metadata, downloading the map, parsing it, and
    processing its tags to create a structured dictionary ready for a dataset.

    Args:
        token (str): The OAuth access token for the osu! API.
        offset (int): The offset in the Echo API database to fetch a map from.

    Returns:
        dict: A dictionary containing the processed beatmap data (ID, title,
              hit_objects, tags).
        None: If any step in the pipeline fails.
    """
    # Step 1: Get beatmap metadata from the Echo API.
    echo_api = EchoOsuAPI(ECHO_TOKEN)
    batch = echo_api.get_batch_beatmaps(batch_size=1, offset=offset)

    if not batch:
        print(f"No more maps available at Echo API offset {offset}.")
        return None

    map_data = batch[0]
    beatmap_id = map_data.get('beatmap_id')
    title = map_data.get('title', 'Unknown Title')
    print(f"Processing: {title} (ID: {beatmap_id})")

    # Step 2: Download the .osu file using the official osu! API.
    osu_file_path = get_beatmap_file(beatmap_id, token)

    if not osu_file_path:
        return None  # Stop processing if the download failed.

    # Step 3: Parse the local .osu file to extract data.
    try:
        parser = OsuFileParser(osu_file_path)
        parser.read_file()
        hit_objects = parser.extract_raw_hit_objects()

        if not hit_objects:
            print("Warning: No hit objects were extracted from the beatmap.")

        # Step 4: Filter the tags from the Echo API using predefined rules.
        api_tags_data = map_data.get('tags', [])
        print(
            f"Original tags from API: {[tag.get('tag') for tag in api_tags_data]}")

        filtered_tags = filter_tags(api_tags_data, tag_counts)
        print(f"Filtered tags after processing: {filtered_tags}")

        # Step 5: Assemble the final training sample dictionary.
        return {
            'beatmap_id': beatmap_id,
            'title': title,
            'hit_objects': hit_objects,
            'tags': filtered_tags
        }

    except Exception as e:
        print(f"An error occurred during file parsing or tag filtering: {e}")
        return None


# This block runs only when the script is executed directly.
if __name__ == "__main__":
    print("--- Starting Single Map Processing Pipeline ---")
    # First, obtain the necessary API token.
    osu_api_token = get_oauth_token()

    # Proceed only if the token was successfully acquired.
    if osu_api_token:
        # Process one map from the Echo API (at offset 0 by default).
        training_sample = process_one_api_map(osu_api_token)
        if training_sample:
            print("\n--- Pipeline Complete ---")
            print(
                f"Successfully created a training sample for: {training_sample['title']}")
            print(
                f"Number of hit objects: {len(training_sample['hit_objects'])}")
            print(f"Final tags: {training_sample['tags']}")
        else:
            print("\n--- Pipeline Failed ---")
            print("Could not create a training sample.")
    else:
        print("Could not start pipeline due to authentication failure.")

