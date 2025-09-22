# echosu_api.py
"""
This module provides a dedicated class, EchoOsuAPI, for interacting with the
echosu.com API. It encapsulates the necessary authorization and request logic
to fetch beatmap tags and other metadata.
"""

import requests


class EchoOsuAPI:
    """
    A wrapper class for the echosu.com API.

    Handles authentication and provides methods to easily access different
    API endpoints related to beatmaps and their tags.

    Attributes:
        base_url (str): The base URL for all API endpoints.
        headers (dict): A dictionary containing the necessary headers for
                        API requests, including the authorization token.
    """

    def __init__(self, api_token):
        """
        Initializes the EchoOsuAPI client.

        Args:
            api_token (str): The secret API token obtained from echosu.com
                             required for authenticating requests.
        """
        self.base_url = 'https://www.echosu.com'
        # The API requires a 'Token' authorization header.
        self.headers = {
            'Authorization': f'Token {api_token}',
            'Content-Type': 'application/json',
        }

    def get_beatmap_tags(self, beatmap_id):
        """
        Fetches all associated tags for a single beatmap by its ID.

        Args:
            beatmap_id (str or int): The unique identifier for the beatmap.

        Returns:
            list: A list of tag dictionaries (e.g., [{'tag': 'aim', 'count': 10}, ...])
                  if the request is successful and tags exist.
            list: An empty list if the request fails, the map has no tags, or an
                  error occurs.
        """
        # Construct the full URL for the specific API endpoint.
        url = f'{self.base_url}/api/beatmaps/{beatmap_id}/tags/'

        try:
            # Perform the GET request to the API.
            response = requests.get(url, headers=self.headers)
            print(
                f"Echo API response for beatmap {beatmap_id}: {response.status_code}")
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

            # The API returns a list containing one dictionary with a 'tags' key.
            data = response.json()
            if data and 'tags' in data[0]:
                return data[0]['tags']
            return []

        except requests.exceptions.RequestException as e:
            print(f"API request failed for beatmap {beatmap_id}: {e}")
            return []

    def get_batch_beatmaps(self, batch_size=50, offset=0):
        """
        Fetches a batch of beatmaps, including their titles and tags.

        This endpoint is useful for discovering maps and building a dataset.

        Args:
            batch_size (int): The number of beatmaps to retrieve in one call.
            offset (int): The starting point in the database to fetch from.
                          Useful for paginating through all available maps.

        Returns:
            list: A list of beatmap dictionaries if the request is successful.
            list: An empty list if the request fails or an error occurs.
        """
        url = f'{self.base_url}/api/beatmaps/tags/'
        params = {
            'batch_size': batch_size,
            'offset': offset
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            print(f"Echo API batch request response: {response.status_code}")
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"API batch request failed: {e}")
            return []


# This block of code runs only when the script is executed directly.
# It serves as a simple test to verify that the API connection is working.
if __name__ == "__main__":
    # WARNING: Do not commit your API token directly into your code.
    # Use environment variables or a configuration file for better security.
    TEST_API_TOKEN = "aCuKVN7vValzoM9U8N1atRAQKo4BWslyMqPQb9YYq7wg89T8T75jK8TsssrSnbia"
    api = EchoOsuAPI(TEST_API_TOKEN)

    # --- Test 1: Get tags for a single, specific beatmap ---
    print("--- Testing single beatmap endpoint ---")
    # Using a known beatmap ID for a consistent test case.
    example_beatmap_id = "2897724"
    tags = api.get_beatmap_tags(example_beatmap_id)
    if tags:
        print(
            f"Tags for beatmap {example_beatmap_id}: {[tag['tag'] for tag in tags]}")
    else:
        print(
            f"No tags found for beatmap {example_beatmap_id} or API call failed.")

    # --- Test 2: Get a small batch of beatmaps ---
    print("\n--- Testing batch retrieval endpoint ---")
    batch = api.get_batch_beatmaps(batch_size=5, offset=1000)
    if batch:
        print(f"Successfully retrieved a batch of {len(batch)} beatmaps:")
        # Print a summary of each beatmap in the retrieved batch.
        for beatmap in batch:
            beatmap_title = beatmap.get('title', 'Unknown Title')
            beatmap_tags = [tag.get('tag') for tag in beatmap.get('tags', [])]
            print(f"  - {beatmap_title} | Tags: {beatmap_tags}")
    else:
        print("Batch retrieval failed.")
