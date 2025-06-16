# run_card_reader.py
import os
import sys
from card_reader_class import CardProcessor
import subprocess
import json

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
    # Extract debug, JSON, and input paths
    debug = "--d" in sys.argv
    to_json = "--j" in sys.argv
    input_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    if not input_args:
        print("Usage: python run_card_reader.py [--d] [--j] <image_or_folder1> <image_or_folder2> ...")
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

    # Process each image using the CardProcessor class
    for img_path in all_image_paths:
        print(f"\n>>> Processing {img_path}")
        cmd = ["python", "card_reader_class.py"]
        if debug:
            cmd.append("--d")
        if to_json:
            cmd.append("--j")
        cmd.append(img_path)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {img_path}:\n{e.stderr}")

if __name__ == "__main__":
    main()
