"""
The official osu! API v2: log in, and download a beatmap's .osu file.

What is left of complete_pipeline.py. Its single-map demo pipeline was removed
(it called an Echo API method that no longer exists); build-dataset
(osu_tagger.data.builder) is the pipeline, and it uses these two functions.
Credentials come from .env: OSU_CLIENT_ID and OSU_CLIENT_SECRET.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()
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
