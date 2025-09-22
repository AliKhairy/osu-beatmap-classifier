# predict_for_overlay.py
"""
This script serves as a simple command-line interface for the beatmap classifier,
designed to be called by an external application, such as a C# overlay.

It takes a single command-line argument: the full path to a .osu beatmap file.
It then loads the trained model, runs a prediction, and prints the resulting tags
to standard output as a single, comma-separated string.

All diagnostic or error messages are printed to standard error to avoid
polluting the primary output stream.

Usage (from an external app):
> python predict_for_overlay.py "C:\\Path\\To\\Your\\beatmap.osu"

Successful Output:
> streams,jumps,tech

Error Output:
> (Printed to stderr) Error: classifier.pkl not found.
"""

import sys
import os
import pickle
# The neural_model module is required because the pickled object is an instance
# of the ImprovedBeatmapClassifier class defined within it.
from neural_model import ImprovedBeatmapClassifier


def main():
    """
    Main execution function for the script.
    """
    # --- Step 1: Read the beatmap path from command-line arguments ---
    # sys.argv is a list of command-line arguments.
    # sys.argv[0] is the script name itself ('predict_for_overlay.py').
    # sys.argv[1] is the first argument passed to the script.
    if len(sys.argv) < 2:
        # Print to stderr so the calling application knows it's an error.
        print("Error: Missing command-line argument. You must provide a path to a .osu file.", file=sys.stderr)
        sys.exit(1)  # Exit with a non-zero status code to indicate failure.

    map_path = sys.argv[1]

    # Allows for an optional second argument for the project's root path.
    # This helps locate the model file if the script is run from a different directory.
    project_path = sys.argv[2] if len(sys.argv) > 2 else '.'
    classifier_path = os.path.join(project_path, 'beatmap_classifier.pkl')

    # --- Step 2: Load the trained classifier ---
    try:
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(
                f"Classifier file not found at '{classifier_path}'")

        with open(classifier_path, 'rb') as f:
            classifier = pickle.load(f)

        # --- Step 3: Run prediction ---
        # A lower threshold might be suitable for an overlay to be more inclusive.
        predicted_tags = classifier.predict_tags(map_path, threshold=0.286)

        # --- Step 4: Print the result to standard output ---
        if predicted_tags and isinstance(predicted_tags, list):
            # On success, print ONLY the comma-separated tags.
            # This makes it easy for the C# app to parse the result.
            print(",".join(predicted_tags))
        else:
            # Handle cases where prediction returns an empty list or an error message.
            print(
                f"Prediction resulted in an error or no tags met the threshold: {predicted_tags}", file=sys.stderr)

    except FileNotFoundError as e:
        print(
            f"Error: {e}. Please ensure the model has been trained.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors during prediction.
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
