# Handwritten fossil card digitization

## How to use
1. Clone the repository
2. Install the required packages with `pip install -r requirements.txt` (creating a virtual environment is good)
3. Download pretrained models for dental OCR from [Riikka](https://github.com/korolainenriikka/fine-tuned-ocr-for-dental-markings/tree/main/pretrained_models). Need to have a folder named `pretrained_models` in this repository with the .pt files in it (If the .pt files get zipped before downloading they may get corrupted, so donwload the raw files individually without zipping).
3. Run the script with `python run_card_reader.py [--d] path/to/image.tif`. The `--d` option enables debug mode, which displays intermediate processing information such as template matching scores and overlay visualizations. The tool also works for .jpg files.


## How to run the Card Reader System:

Command:
```bash
python run_card_reader.py [--d] <path_to_image1> <path_to_image2> ....
```
- <path_to_imageX>: One or more image paths to process.  (.tif, .png, .jpg are supported)
- --d: (Optional) enables debug mode, which displays intermediate processing information
such as template matching scores and overlay visualizations.

The tool can also handle folders. If you provide a folder path, it will process all images in that folder and its subfolders.

The output is saved to a .CSV file named `output.csv` in the current working directory. The CSV file contains the following columns:
- `Image path`: The path to the processed image.
- `Card type`: The type of card detected (e.g., "accbig", "accprint", etc.).
- `Dental markings`: Yes or No, indicating whether dental markings were detected.
- `Key-value pairs`: A string representation of the key-value pairs extracted from the card and related confidence scores.

OR

Import the CardProcessor class into another python file/program and call it's card_reader function with an image path.

