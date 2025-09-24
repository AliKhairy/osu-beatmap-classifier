# main.py (Final Version with Offset)
"""
Main orchestrator for the osu! Beatmap Classifier.
Handles dataset building, model training, and predictions.
"""
import os
import pickle
from dataset_builder import build_full_dataset
from neural_model import ImprovedBeatmapClassifier
from rebuild_from_downloaded import rebuild

# The number of maps to attempt to download when building a new dataset.
DATA_SET_SIZE = 1800


def main():
    """
    Main function to run the application CLI.
    Checks for existing datasets and models and provides user options.
    """
    print("osu! Beatmap Classifier")
    print("=" * 50)

    has_dataset = os.path.exists('ml_dataset.json')
    classifier = ImprovedBeatmapClassifier()
    has_model = classifier.model_exists()

    if has_dataset and has_model:
        print("\nFound existing dataset and trained model.")
        choice = input(
            "Select an option:\n"
            "1. Use existing model for predictions\n"
            "2. Retrain model with the current dataset\n"
            "3. Rebuild dataset and retrain model\n"
            "4. Test model on multiple maps\n"
            "Choice (1-4): "
        )
        if choice == '1' or choice == '4':
            print("Loading existing model...")
            pass
        elif choice == '2':
            print("Retraining model with existing dataset...")
            classifier.train()
        elif choice == '3':
            print("Rebuilding dataset and model...")
            cleanup_files()
            
            # Ask the user for a start offset.
            try:
                offset_input = input("Enter start offset (or leave blank for 0): ")
                start_offset = int(offset_input) if offset_input else 0
            except ValueError:
                start_offset = 0
            
            # Pass the offset to the build function.
            build_full_dataset(max_maps=DATA_SET_SIZE, offset=start_offset)
            classifier.train()
        elif choice == '4':
            classifier.test_multiple_maps(max_maps=10, threshold=0.25)
            return

    elif has_dataset:
        print("\nDataset found. Training a new model...")
        classifier.train()

    else:
        print("\nNo dataset or model found. Starting initial build...")
        # Also ask for an offset on the very first build.
        try:
            offset_input = input("Enter start offset (or leave blank for 0): ")
            start_offset = int(offset_input) if offset_input else 0
        except ValueError:
            start_offset = 0
            
        build_full_dataset(max_maps=DATA_SET_SIZE, offset=start_offset)
        rebuild()
        classifier.train()

    prediction_loop(classifier)


def prediction_loop(classifier):
    """Handles the interactive prediction menu."""
    print("\n--- Prediction Mode ---")
    while True:
        print("\nOptions:")
        print("1. Predict tags for a single map")
        print("2. Test model on multiple maps")
        print("3. Quit")

        choice = input("Choice (1-3): ")

        if choice == '1':
            predict_single_map(classifier)
        elif choice == '2':
            try:
                threshold = float(input("Enter prediction threshold (e.g., 0.27): ") or "0.27")
                max_maps = int(input("Enter number of maps to test (e.g., 5): ") or "5")
                classifier.test_multiple_maps(max_maps=max_maps, threshold=threshold)
            except ValueError:
                print("Invalid input. Please enter a number.")
        elif choice == '3':
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def predict_single_map(classifier):
    """Guides the user through selecting and predicting a single beatmap."""
    songs_folder = "songs"
    if not os.path.exists(songs_folder):
        print(f"Error: Songs folder not found at '{songs_folder}'")
        return

    osu_files = [f for f in os.listdir(songs_folder) if f.endswith('.osu')]
    if not osu_files:
        print(f"Error: No .osu files found in '{songs_folder}'")
        return

    print(f"\nAvailable maps in '{songs_folder}':")
    for i, file in enumerate(osu_files[:20]):
        display_name = file if len(file) <= 70 else file[:67] + "..."
        print(f"  {i+1:2d}. {display_name}")

    if len(osu_files) > 20:
        print(f"  ... and {len(osu_files) - 20} more")

    try:
        choice = input(f"\nEnter map number (1-{len(osu_files)}) or filename: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(osu_files):
            chosen_file = osu_files[int(choice) - 1]
        else:
            chosen_file = choice if choice.endswith('.osu') else choice + '.osu'
        
        map_path = os.path.join(songs_folder, chosen_file)

        if os.path.exists(map_path):
            threshold = float(input("Enter prediction threshold (e.g., 0.27): ") or "0.27")
            predicted_tags = classifier.predict_tags(map_path, threshold=threshold)
            print(f"\nFinal prediction: {predicted_tags}")
        else:
            print(f"Error: File not found at '{map_path}'")
    except (ValueError, IndexError):
        print("Invalid selection.")


def cleanup_files():
    """Removes generated dataset and model files for a clean rebuild."""
    files_to_remove = [
        'ml_dataset.json',
        'beatmap_classifier.pkl'
    ]
    print("Removing old generated files...")
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed {file}")


if __name__ == "__main__":
    main()