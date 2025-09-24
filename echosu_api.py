# echosu_api.py
"""
This module provides a dedicated class, EchoOsuAPI, for interacting with the
echosu.com API.
"""
import requests


class EchoOsuAPI:
    def __init__(self, api_token):
        self.base_url = 'https://www.echosu.com'
        self.headers = {
            'Authorization': f'Token {api_token}',
            'Content-Type': 'application/json',
        }

    def discover_all_beatmaps(self):
        """
        Fetches the entire list of beatmaps in a single request.
        """
        url = f'{self.base_url}/api/beatmaps/'
        print(f"Fetching the full list of beatmaps from {url}...")
        try:
            # We set a long timeout because this request could take a while.
            response = requests.get(url, headers=self.headers, timeout=60)
            print(f"Echo API request response: {response.status_code}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return []

    def get_beatmap_tags(self, beatmap_id):
        """
        Fetches all tags for a single beatmap.
        """
        url = f'{self.base_url}/api/beatmaps/{beatmap_id}/'
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            tags_list = data.get('tags', [])
            if tags_list:
                return [{"tag": tag['name']} for tag in tags_list]
            return []
        except requests.exceptions.RequestException as e:
            return []
