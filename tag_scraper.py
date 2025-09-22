# tag_scraper.py
"""
This module is responsible for scraping the tag library from echosu.com.

It fetches the complete list of all possible tags and their usage counts.
This data is then used to filter out unpopular or irrelevant tags from the
beatmaps in our dataset, ensuring the model is trained only on meaningful labels.
"""

import requests
from bs4 import BeautifulSoup


def scrape_tag_counts(url="https://echosu.com/tag_library/"):
    """
    Scrapes the echosu.com tag library to get a dictionary of tags and their counts.

    Args:
        url (str): The URL of the tag library page.

    Returns:
        dict: A dictionary mapping each tag name (str) to its usage count (int).
              Returns an empty dictionary if scraping fails.
    """
    print(f"Scraping tag library from {url}...")
    try:
        # Perform an HTTP GET request to the specified URL.
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes.

        # Parse the HTML content of the page using BeautifulSoup.
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the unordered list element with the specific ID 'tag-list'.
        tag_list = soup.find('ul', id='tag-list')
        if not tag_list:
            print("Error: Could not find the 'tag-list' element on the page.")
            return {}

        tag_counts = {}
        # Iterate over each list item ('li') within the tag list.
        for item in tag_list.find_all('li'):
            tag_name_span = item.find('span', class_='tag-name')
            tag_count_span = item.find('span', class_='tag-count')

            if tag_name_span and tag_count_span:
                name = tag_name_span.text.strip()
                # Extract the count text (e.g., "123 maps").
                count_text = tag_count_span.text.strip()

                try:
                    # Isolate the numerical part and convert it to an integer.
                    count = int(count_text.split()[0])
                    tag_counts[name] = count
                except (ValueError, IndexError):
                    # Handle cases where the count is not a number (e.g., "No maps").
                    tag_counts[name] = 0

        print(f"Successfully scraped {len(tag_counts)} tags.")
        return tag_counts

    except requests.exceptions.RequestException as e:
        print(f"Error: Could not fetch the tag library page. {e}")
        return {}


def filter_tags(api_tags, all_tag_counts):
    """
    Filters a list of tags for a beatmap based on predefined rules and usage counts.

    The goal is to remove niche, irrelevant, or subjective tags to improve the
    quality of the training labels.

    Args:
        api_tags (list): A list of tag dictionaries from the Echo API for a single map.
                         Example: [{'tag': 'aim', 'count': 10}, ...]
        all_tag_counts (dict): A dictionary of all tags and their total usage counts,
                               as scraped from the tag library.

    Returns:
        list: A list of filtered tag names (strings).
    """
    # This list ensures certain important (but potentially low-usage) tags are always kept.
    manual_include = [
        'dense', 'geometric', 'linear patterns',
        'linear aim', 'slider snap', 'berserk jumps'
    ]

    # This list removes common but often ambiguous or unhelpful tags, regardless of their popularity.
    manual_exclude = [
        'marathon', 'farm', 'unconventional farm',
        'dt farm', 'classic'
    ]

    filtered = []
    # The minimum number of times a tag must have been used sitewide to be included.
    min_usage_threshold = 50

    for tag_info in api_tags:
        tag_name = tag_info['tag']

        # Rule 1: Immediately skip any tag that is in the exclusion list.
        if tag_name in manual_exclude:
            continue

        # Rule 2: Check if the tag meets the usage threshold OR is in the manual include list.
        total_count = all_tag_counts.get(tag_name, 0)
        if tag_name in manual_include or total_count >= min_usage_threshold:
            filtered.append(tag_name)

    return filtered


# This block runs only when the script is executed directly.
# It scrapes the data and stores it in a global variable for other modules to import.
tag_counts = scrape_tag_counts()

if not tag_counts:
    print("Warning: Tag counts dictionary is empty. Tag filtering may not work as expected.")
