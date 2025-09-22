# osu_parser.py
"""
This module provides a simple and efficient parser for osu! beatmap files (.osu).
The OsuFileParser class can read a .osu file, extract its metadata, and convert
the raw text of hit objects into a structured format for analysis.
"""


class OsuFileParser:
    """
    Parses a .osu file to extract metadata and hit object data.

    This class reads the contents of a .osu file and provides methods to
    access different sections, such as [Metadata] and [HitObjects].

    Attributes:
        filepath (str): The path to the .osu file.
        lines (list): A list of all lines read from the file.
    """

    def __init__(self, filepath):
        """
        Initializes the OsuFileParser with the path to a beatmap file.

        Args:
            filepath (str): The full path to the .osu file to be parsed.
        """
        self.filepath = filepath
        self.lines = []

    def read_file(self):
        """
        Reads the content of the .osu file into the 'lines' attribute.

        Returns:
            list: The lines of the file. Returns an empty list if the file
                  cannot be found or read.
        """
        try:
            # Open with 'utf-8' encoding, which is standard for .osu files.
            with open(self.filepath, 'r', encoding='utf-8') as file:
                self.lines = file.readlines()
            return self.lines
        except FileNotFoundError:
            print(f"Error: Could not find file at {self.filepath}")
            return []
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return []

    def get_metadata(self):
        """
        Extracts key-value metadata from the [Metadata] section.

        This method parses lines like "Title:My Song" into a dictionary.
        It stops when it reaches a section header (e.g., "[HitObjects]").

        Returns:
            dict: A dictionary containing the beatmap's metadata.
        """
        metadata = {}
        for line in self.lines:
            line = line.strip()
            # Stop parsing if a new section is encountered.
            if line.startswith('['):
                # A small optimization: stop after [Difficulty] as metadata is usually at the top.
                if line.lower() in ["[difficulty]", "[events]", "[timingpoints]", "[hitobjects]"]:
                    break
            # Metadata lines are simple "Key:Value" pairs.
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
        return metadata

    def find_hit_objects_section(self):
        """
        Locates and returns the lines belonging to the [HitObjects] section.

        Returns:
            list: A list of strings, where each string is a raw line from the
                  [HitObjects] section. Returns an empty list if the section
                  is not found.
        """
        try:
            # Find the starting index of the [HitObjects] section.
            start_index = self.lines.index("[HitObjects]\n") + 1
            return self.lines[start_index:]
        except ValueError:
            # This error occurs if "[HitObjects]\n" is not found in the list.
            print(
                f"Warning: [HitObjects] section not found in {self.filepath}")
            return []

    def extract_raw_hit_objects(self):
        """
        Parses the raw text from the [HitObjects] section into a structured list.

        Each hit object is represented as a list containing its core properties.
        This method handles circles, sliders, and spinners.

        Returns:
            list: A list of lists, where each inner list represents one hit object.
                  Format for sliders: [x, y, time, type, curve_type, curve_points, slides, length]
                  Format for circles/spinners: [x, y, time, type, None, [], 1, 0]
        """
        hit_object_lines = self.find_hit_objects_section()
        raw_objects = []

        for line in hit_object_lines:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            parts = line.split(',')

            try:
                # Core properties common to all hit objects.
                x = int(parts[0])
                y = int(parts[1])
                time = int(parts[2])
                type_bit = int(parts[3])

                # Default values for non-slider objects.
                curve_type, curve_points, slides, length = None, [], 1, 0.0

                # Check if the object is a slider (type bit 1 is set).
                is_slider = type_bit & 2
                if is_slider and len(parts) >= 8:
                    # A slider has additional properties defined in subsequent parts.
                    slider_data = parts[5]
                    curve_parts = slider_data.split('|')
                    # B (Bézier), C (Centripetal Catmull-Rom), L (Linear), or P (Perfect circle)
                    curve_type = curve_parts[0]

                    # Parse the slider's anchor points.
                    points_str = curve_parts[1:]
                    curve_points = [[int(p.split(':')[0]), int(
                        p.split(':')[1])] for p in points_str]

                    slides = int(parts[6])
                    length = float(parts[7])

                # Append the structured hit object data.
                raw_objects.append(
                    [x, y, time, type_bit, curve_type, curve_points, slides, length])

            except (ValueError, IndexError) as e:
                print(
                    f"Warning: Skipping malformed hit object line in {self.filepath}: '{line}'. Error: {e}")
                continue

        return raw_objects

    def get_beatmap_id(self):
        """
        A convenience method to get the BeatmapID from the metadata.

        Returns:
            str: The BeatmapID as a string.
            None: If the BeatmapID is not found in the metadata.
        """
        metadata = self.get_metadata()
        return metadata.get('BeatmapID')
