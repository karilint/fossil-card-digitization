
import sys
import os
import json
import cv2
import torch
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from fuzzywuzzy import fuzz, process
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr
from torchvision import transforms


class CardProcessor:
    def __init__(self, debug=False, to_json=False):
        torch.manual_seed(2)
        self.debug = debug
        self.to_json = to_json

        self.debug_print("Loading models and data...")

        self.ocr_model = PaddleOCR(
            lang="en",
            det_algorithm="DB",
            rec_algorithm="SVTR_LCNet",
            det_db_thresh=0.3,
            det_db_box_thresh=0.3,
            invert=True,
            binarize=True,
            drop_score=0.4,
            cls=False,
            show_log=False,
        ) 
        self.shelf_df = pd.read_csv("./Autocorrect/shelf_curated.txt", header=None)[0].to_list() # Key Fields: SHELF
        #self.body_part_df = pd.read_excel("./Autocorrect/BodyElements.xlsx", header=None)[0].to_list()
        self.taxonomy_df = pd.read_csv("./Autocorrect/taxonomy_curated.txt", header=None)[0].to_list() # Key Fields: TAXON, TRIBE, FAMILY, SUB-FAMILY, GENUS, SPECIES
        self.locality_df = pd.read_csv("./Autocorrect/locality_curated.txt", header=None)[0].to_list() # Key Fields: LOCALITY
        self.geography_df = pd.read_csv("./Autocorrect/geography_curated.txt", header=None)[0].to_list() # Key Fields: LOCALITY, SITE, AREA, HORIZON
        
        self.yolo_model = YOLO("runs/detect/train2/weights/best.pt")

        self.dental_models = {
            "uplow": torch.load("pretrained_models/upperlower.pt"),
            "mpi": torch.load("pretrained_models/MPI.pt"),
            "123": torch.load("pretrained_models/123.pt"),
            "1234": torch.load("pretrained_models/1234.pt"),
        }

        self.dental_class_to_idx = {
            "123": {"1": 0, "2": 1, "3": 2},
            "mpi": {"I": 0, "M": 1, "P": 2},
            "1234": {"1": 0, "2": 1, "3": 2, "4": 3},
            "uplow": {"low": 0, "up": 1},
        }

        self.dental_preprocess = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
        ])

    def debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def detect_dentals(self, image_path):
        results = self.yolo_model(image_path)
        markings = []

        # crop and save the detected objects
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Crop the image using the bounding box coordinates
                image = Image.open(image_path).crop((x1, y1, x2, y2)).convert("RGB")

                # OCR the marking
                marking, final_confidence = self.dental_ocr(image)
                self.debug_print(f"Detected marking: {marking}, Confidence: {final_confidence:.4f}")
                markings.append((marking, final_confidence, (x1, y1, x2, y2)))
        return markings

    def dental_ocr(self, cropped_image):
        def infer(model, input_batch):
            with torch.no_grad():
                output = model(input_batch)
                _, prediction = torch.max(output, 1)
                confidence = output.max()
            return prediction, confidence

        def image_to_batch(image):
            return self.dental_preprocess(image).unsqueeze(0)

        def classify(model_key):
            prediction_id, confidence = infer(self.dental_models[model_key], image_to_batch(cropped_image))
            return next(k for k, v in self.dental_class_to_idx[model_key].items() if v == prediction_id), confidence

        uplow, uplow_conf = classify("uplow")
        mpi, mpi_conf = classify("mpi")
        index, index_conf = classify("1234" if mpi == "P" else "123")

        letter = mpi.upper() if uplow == "up" else mpi.lower()
        return f"{letter}{index}", min(uplow_conf, mpi_conf, index_conf).item()

    def overlay_dental_markings(self, image_path, markings):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("doc/fonts/latin.ttf", 100)

        for mark, conf, (x1, y1, x2, y2) in markings:
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=10)
            
            # Write marking and confidence
            draw.text((x1, y1 - 100), f"{mark} ({conf:.3f})", fill="blue", font=font)
        
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    def card_type(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
    
        templates = {
            "accsmalldark": "templates/smalltemplate3.tif",
            "accsmalllight": "templates/lightsmalltemplate.tif",
            "accbig": "templates/bigtemplate2.tif",
            "accprint": "templates/printtemplate2.tif",
            "accbox1": "templates/boxtemplate1.png",
            "accbox2": "templates/boxtemplate2.png"
        }
    
        scores = {}
        for k, v in templates.items():
            template_gray = cv2.imread(v, cv2.IMREAD_GRAYSCALE)
            if template_gray is None:
                print(f"Warning: failed to read template {v}")
                continue
            
            # Resize template if larger than input image
            if (template_gray.shape[0] > image.shape[0]) or (template_gray.shape[1] > image.shape[1]):
                scale_y = image.shape[0] / template_gray.shape[0]
                scale_x = image.shape[1] / template_gray.shape[1]
                scale = min(scale_x, scale_y)
                new_size = (int(template_gray.shape[1] * scale), int(template_gray.shape[0] * scale))
                template_gray = cv2.resize(template_gray, new_size)
    
            try:
                score = float(np.max(cv2.matchTemplate(image, template_gray, cv2.TM_CCOEFF_NORMED)))
                scores[k] = score
            except cv2.error as e:
                self.debug_print(f"Template matching failed for {k} due to: {e}")
                scores[k] = 0.0
    
        self.debug_print("Card type match scores:", scores)
        return max(scores, key=scores.get) if scores else None

    def clean_dot_lines(self, image_path, card_type):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if card_type == "accsmalldark":
            template_path = "templates/template5.tif"
        elif card_type == "accbox1":
            template_path = "templates/boxtemplate.png"
        else:
            template_path = "templates/lighttemplate.tif"
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        color = np.mean(image[:100, -100:], axis=(0, 1))
        for pt in zip(*loc[::-1]):
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), tuple(map(int, color)), -1)
        cv2.imwrite("temp/temp.tif", image)

    def sort_results(self, results):
        def get_y_overlap(box1, box2):
            top1, bottom1 = box1[0][1], box1[3][1]  # y-coordinates of first box
            top2, bottom2 = box2[0][1], box2[3][1]  # y-coordinates of second box

            overlap = max(0, min(bottom1, bottom2) - max(top1, top2))
            height1, height2 = bottom1 - top1, bottom2 - top2

            return overlap / min(
                height1, height2
            )  # Compute overlap percentage relative to the smaller box

        # first sort results by x-coordinates
        results = sorted(results, key=lambda x: x[0][0][0])

        n = len(results)
        swapped = True
        while swapped:
            swapped = False
            for i in range(n - 1):
                box1, box2 = results[i][0], results[i + 1][0]
                overlap = get_y_overlap(box1, box2)

                if overlap >= 0.4:
                    # Same line, check x-coordinates
                    if box1[0][0] > box2[0][0]:
                        results[i], results[i + 1] = results[i + 1], results[i]
                        swapped = True
                else:
                    # Different lines, check y-coordinates
                    if box1[0][1] > box2[0][1]:
                        results[i], results[i + 1] = results[i + 1], results[i]
                        swapped = True
        return results

    def overlay_ocr_results(self, image_path, results):
        image = cv2.imread(image_path)

        # Convert BGR to RGB for proper display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        # Extract bounding boxes, text, and confidence scores
        boxes = [r[0] for r in results]
        texts = [r[1][0] for r in results]
        scores = [r[1][1] for r in results]

        # Draw OCR results on the image
        annotated = draw_ocr(image, boxes, texts, scores, font_path="doc/fonts/simfang.ttf")

        # Display the image
        plt.figure(figsize=(20, 10))
        plt.imshow(annotated)
        plt.axis("off")
        plt.show()

    def key_value(self, results, w, h, card_type):
        if card_type in ["accsmalldark", "accsmalllight"]:
            keys = {
                "Shelf": [],
                "Accession No.": [],
                "Field No.": [],
                "Taxon": [],
                "Part": [],
                "Locality": [],
                "Site": [],
                "Horizon": [],
                "Additional": [],
            }

            key_phrases = [
                "Accession No.",
                "Field No.",
                "Taxon",
                "Part",
                "Locality",
                "Site",
                "Horizon"
            ]

            current_key = None
            for item in results:
                text, score = item[1]
                x, y = item[0][0]  # Top-left corner

                if x > 0.861 * w and y < 0.1695 * h:
                    keys["Shelf"].append((text, score))
                    continue
                
                # Check if it belongs in Additional
                if y > h / 2 or x > 0.8699 * w:
                    keys["Additional"].append((text, score))
                    continue
                
                # Assign based on recognized key words and remove the key phrase from text
                for phrase in key_phrases:
                    if len(text) >= len(phrase):
                        ratio = fuzz.ratio(phrase, text[:len(phrase)])
                        if ratio >= 70:
                            current_key = phrase
                            text = text[len(phrase):].strip()
                            break
                    else:
                        ratio = fuzz.ratio(phrase, text)
                        if ratio >= 70:
                            current_key = phrase
                            text = ""
                            break

                if current_key and text:
                    keys[current_key].append((text, score))
            return keys

        elif card_type == "accbig":
            keys = {
                "SHELF:": [],
                "FAMILY:": [],
                "SUB-FAMILY:": [],
                "GENUS:": [],
                "SPECIES:": [],
                "NATURE OF SPECIMEN:": [],
                "LOCALITY:": [],
                "SITE:": [],
                "ACC No.:": [],
                "FIELD No.:": [],
                "ADDITIONAL:": [],
            }

            key_phrases = [
                "FAMILY:",
                "SUB-FAMILY:",
                "GENUS:",
                "SPECIES:",
                "NATURE OF SPECIMEN:",
                "LOCALITY:",
                "SITE:",
                "ACC No.:",
                "FIELD No.:",
            ]

            current_key = None
            for item in results:
                text, score = item[1]
                x, y = item[0][0] # Top-left corner

                if x > 0.8756 * w and y < 0.1274 * h:
                    keys["SHELF:"].append((text, score))
                    continue

                # Check if it belongs in Additional
                if y > 0.4982 * h or x > 0.8415 * w:
                    keys["ADDITIONAL:"].append((text, score))
                    continue
                
                # Assign based on recognized key words and remove the key phrase from text
                for phrase in key_phrases:
                    if len(text) >= len(phrase):
                        ratio = fuzz.ratio(phrase, text[:len(phrase)])
                        if ratio >= 70:
                            current_key = phrase
                            text = text[len(phrase):].strip()
                            break
                    else:
                        ratio = fuzz.ratio(phrase, text)
                        if ratio >= 70:
                            current_key = phrase
                            text = ""
                            break

                if current_key and text:
                    keys[current_key].append((text, score))
            return keys

        elif card_type == "accbox1":
            keys = {
                "Shelf": [],
                "Accession No.": [],
                "Field No.": [],
                "Taxon": [],
                "Part": [],
                "Locality": [],
                "Site": [],
                "Horizon": [],
                "GPS X": [],
                "Additional": [],
            }

            key_phrases = [
                "Accession No.",
                "Field No.",
                "Taxon",
                "Part",
                "Locality",
                "Site",
                "Horizon",
                "GPS X"
            ]

            current_key = None
            for item in results:
                text, score = item[1]
                x, y = item[0][0]  # Top-left corner

                if x > 0.7742 * w and y < 0.1095 * h:
                    keys["Shelf"].append((text, score))
                    continue
                
                # Check if it belongs in Additional
                if y > h / 2:
                    keys["Additional"].append((text, score))
                    continue
                
                # Assign based on recognized key words and remove the key phrase from text
                for phrase in key_phrases:
                    if len(text) >= len(phrase):
                        ratio = fuzz.ratio(phrase, text[:len(phrase)])
                        if ratio >= 70:
                            current_key = phrase
                            text = text[len(phrase):].strip()
                            break
                    else:
                        ratio = fuzz.ratio(phrase, text)
                        if ratio >= 70:
                            current_key = phrase
                            text = ""
                            break

                if current_key and text:
                    keys[current_key].append((text, score))
            return keys
        
        elif card_type == "accbox2":
            keys = {
                "Shelf No:": [],
                "FAMILY:": [],
                "SUB-FAMILY:": [],
                "GPS": [],
                "GENUS:": [],
                "SPECIES:": [],
                "NATURE OF SPECIMEN:": [],
                "LOCALITY:": [],
                "SITE:": [],
                "ACC No.:": [],
                "FIELD No.:": [],
                "ADDITIONAL:": [],
            }

            key_phrases = [
                "FAMILY:",
                "SUB-FAMILY:",
                "GPS",
                "GENUS:",
                "SPECIES:",
                "NATURE OF SPECIMEN:",
                "LOCALITY:",
                "SITE:",
                "ACC No.:",
                "FIELD No.:",
            ]

            current_key = None
            for item in results:
                text, score = item[1]
                x, y = item[0][0] # Top-left corner

                if x > 0.7517 * w and y < 0.1204 * h:
                    keys["Shelf No:"].append((text, score))
                    continue

                if 0.6311 * w < x < 0.9079 * w and 0.1872 * h < y < 0.2996 * h:
                    keys["GPS"].append((text, score))
                    continue

                # Check if it belongs in Additional
                if y > 0.4982 * h:
                    keys["ADDITIONAL:"].append((text, score))
                    continue
                
                # Assign based on recognized key words and remove the key phrase from text
                for phrase in key_phrases:
                    if len(text) >= len(phrase):
                        ratio = fuzz.ratio(phrase, text[:len(phrase)])
                        if ratio >= 70:
                            current_key = phrase
                            text = text[len(phrase):].strip()
                            break
                    else:
                        ratio = fuzz.ratio(phrase, text)
                        if ratio >= 70:
                            current_key = phrase
                            text = ""
                            break

                if current_key and text:
                    keys[current_key].append((text, score))
            return keys

        elif card_type == "accprint":
            keys = {
                "Shelf No.": [],
                "Accession No.": [],
                "Field No.": [],
                "Taxon": [],
                "Tribe": [],
                "Genus": [],
                "Species": [],
                "Body part": [],
                "Area": [],
                "Horizon": [],
                "Year": [],
                "Aerial photo": [],
                "Additional": [],
            }

            key_phrases = [
                "Accession No.",
                "Field No.",
                "Taxon",
                "Tribe",
                "Genus",
                "Species",
                "Body part",
                "Area",
                "Horizon",
                "Year",
                "Aerial photo"
            ]

            current_key = None
            for item in results:
                text, score = item[1]
                x, y = item[0][0]  # Top-left corner

                if x > 0.8308 * w and y < 0.1093 * h:
                    keys["Shelf No."].append((text, score))
                    continue
                
                # Check if it belongs in Additional
                if y > h / 2:
                    keys["Additional"].append((text, score))
                    continue
                
                # Assign based on recognized key words and remove the key phrase from text
                for phrase in key_phrases:
                    if len(text) >= len(phrase):
                        ratio = fuzz.ratio(phrase, text[:len(phrase)])
                        if ratio >= 70:
                            current_key = phrase
                            text = text[len(phrase):].strip()
                            break
                    else:
                        ratio = fuzz.ratio(phrase, text)
                        if ratio >= 70:
                            current_key = phrase
                            text = ""
                            break

                if current_key and text:
                    keys[current_key].append((text, score))
            return keys
        
        else:
            return None
        
        return {}

    # This function normalizes the OCR dictionary by uppercasing all string keys and values and ensuring keys end with a colon
    def normalize_ocr_dict(self, keys):
        normalized_dict = {}

        for key, value_list in keys.items():

            # Uppercase the key and ensure it ends with a colon
            new_key = key.upper()
            if not new_key.endswith(":"):
                new_key += ":"
            
            # Process each (text, confidence) tuple in the value list
            new_values = []
            for item in value_list:
                if isinstance(item, tuple) and isinstance(item[0], str):
                    upper_text = item[0].upper()
                    new_values.append((upper_text, item[1]))
                else:
                    new_values.append(item)  # leave non-tuple or non-string values untouched

            normalized_dict[new_key] = new_values
        return normalized_dict

    def clean_empty_entries(self, ocr_dict):
        cleaned = {}
        for key, values in ocr_dict.items():
            new_values = []
            for text, conf in values:
                # strip whitespace/punctuation, then check for alphanumeric
                stripped = text.strip(string.whitespace + string.punctuation)
                if stripped and any(ch.isalnum() for ch in stripped):
                    new_values.append((text, conf))
            if new_values:
                cleaned[key] = new_values
        return cleaned

    def autocorrect_ocr_fields(self, keys, threshold=None):
        corrected = keys.copy()
    
        # Define fields that should always be corrected regardless of score
        ignore_threshold_fields = {"SHELF:"}
    
        def correct_field(field_name, candidate_list):
            if field_name in corrected and corrected[field_name]:
                updated = []
                for text, conf in corrected[field_name]:
                    match, score = process.extractOne(text, candidate_list)
                    if field_name in ignore_threshold_fields:
                        # Always correct, even if score is low
                        updated.append((match, score))
                    elif threshold is None or score >= threshold * 100:
                        updated.append((match, score))
                    else:
                        updated.append((text, conf))
                corrected[field_name] = updated
    
        # Apply correction
        correct_field("SHELF:", self.shelf_df)

        correct_field("TAXON:", self.taxonomy_df)
        correct_field("TRIBE:", self.taxonomy_df)
        correct_field("FAMILY:", self.taxonomy_df)
        correct_field("SUB-FAMILY:", self.taxonomy_df)
        correct_field("GENUS:", self.taxonomy_df)
        correct_field("SPECIES:", self.taxonomy_df)

        correct_field("LOCALITY:", self.locality_df) # locality has its own list compared to the rest since its values are clearly defined
        correct_field("SITE:", self.geography_df)
        correct_field("AREA:", self.geography_df)
        correct_field("HORIZON:", self.geography_df)

        #correct_field("PART:", self.body_part_df)
        #correct_field("BODY PART:", self.body_part_df)
        #correct_field("NATURE OF SPECIMEN:", self.body_part_df)

        return corrected

    def save_to_file(self, image_path, card_type, dental_markings, key_value_dict):
        # change the path to desired output path
        file_path = "output.csv"

        # Flatten key-value dictionary into a single string
        kv_pairs = [
            f"{key}: {value}" for key, value in key_value_dict.items()
        ]
        kv_string = "; ".join(kv_pairs)

        result = {
            "Image path": image_path,
            "Card type": card_type,
            "Dental markings": "Yes" if dental_markings else "No",
            "Key-value pairs": kv_string
        }

        # Create new data row as a DataFrame
        new_row = pd.DataFrame([result])

        # Append or create the CSV file
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            updated_df.to_csv(file_path, index=False)
        else:
            new_row.to_csv(file_path, index=False)

        if self.to_json:
            print("\n--- Converting to JSON output ---")
            return json.dumps(result)

    def card_reader(self, image_path):
        input_img_path = image_path
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        w, h = image.shape[::-1]

        self.debug_print("--- Dental Marking Detection ---")
        markings = self.detect_dentals(image_path)
        if self.debug:
            self.overlay_dental_markings(image_path, markings)

        self.debug_print("--- Card Type Detection ---")
        toc = self.card_type(image_path)
        self.debug_print("Card type:", toc)

        if toc in ["accsmalldark", "accsmalllight", "accbox1"]:
            self.debug_print("\n--- Cleaning Dot Lines ---")
            self.clean_dot_lines(image_path, toc)
            image_path = "temp/temp.tif"

        self.debug_print("\n--- OCR with PaddleOCR ---")
        results = self.ocr_model.ocr(image_path, cls=False)[0]
        self.debug_print("\n --- Sorting OCR Results ---")
        sorted_results = self.sort_results(results)
        self.debug_print(sorted_results)
        self.debug_print("\n--- Overlaying OCR Results ---")
        if self.debug:
            self.overlay_ocr_results(image_path, sorted_results)

        self.debug_print("\n--- Key-Value Extraction ---")
        keys = self.key_value(sorted_results, w, h, toc)
        U_keys = self.normalize_ocr_dict(keys)
        U_keys = self.clean_empty_entries(U_keys)
        self.debug_print(U_keys)

        self.debug_print("\n--- Autocorrected Key-Value Pairs ---")
        keys_corr = self.autocorrect_ocr_fields(U_keys, threshold=0.7)

        self.debug_print("\n--- add dental markings to output dental markings ---")
        keys_corr["DENTAL_MARKINGS:"] = [(mark, round(conf, 3)) for mark, conf, _  in markings]

        #for key, values in keys_corr.items():
        #    print(key)
        #    for val in values:
        #        print(val)
        #    print()

        # delete temporary image file if it exists
        if os.path.exists("temp/temp.tif"):
            os.remove("temp/temp.tif")  

        self.debug_print("\n--- Saving to output file ---")
        if self.to_json:
            return self.save_to_file(input_img_path, toc, bool(markings), keys_corr)
        else:
            self.save_to_file(input_img_path, toc, bool(markings), keys_corr)


if __name__ == "__main__":
    debug = "--d" in sys.argv
    to_json = "--j" in sys.argv
        
    # Filter out flags from image paths
    image_paths = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    if image_paths:
        processor = CardProcessor(debug=debug, to_json=to_json)
        for img_path in image_paths:
            print(f"\n==== Processing {img_path} ====")
            result = processor.card_reader(img_path)
            #if result and to_json:
            #    try:
            #        json_output = json.loads(result)
            #        print(json.dumps(json_output))
            #    except (TypeError, ValueError) as e:
            #        print("Error: Output is not valid JSON.", e)
    else:
        print("Usage: python run_card_reader.py [--d] [--j] <image_or_folder1> <image_or_folder2> ...")
        sys.exit(1)
