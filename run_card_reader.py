# run_card_reader.py
import os
import sys
from card_reader_class import CardProcessor

def collect_images_from_path(path):
    """Return a list of valid image file paths from a given path."""
    valid_extensions = {'.tif', '.png', '.jpg', '.jpeg'}
    image_paths = []

    if os.path.isfile(path):
        if os.path.splitext(path)[1].lower() in valid_extensions:
            image_paths.append(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    image_paths.append(os.path.join(root, file))

    return image_paths

def main():
    # Extract debug flag and input paths
    debug = "--d" in sys.argv
    input_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    if not input_args:
        print("Usage: python run_card_reader.py [--d] <image_or_folder1> <image_or_folder2> ...")
        return

    # Collect all valid image paths
    all_image_paths = []
    for arg in input_args:
        all_image_paths.extend(collect_images_from_path(arg))

    # sort image paths to ensure consistent processing order
    all_image_paths = sorted(set(all_image_paths))  

    if not all_image_paths:
        print("No valid image files found.")
        return

    # Initialize processor
    processor = CardProcessor(debug=debug)

    for img_path in all_image_paths:
        print(f"\n>>> Processing {img_path}")
        processor.card_reader(img_path)

if __name__ == "__main__":
    main()