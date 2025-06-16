# Requirements Documentation

This document describes the purpose of each package listed in `requirements.txt` and how it is used in `card_reader_class.py`.

---

## 1. **fuzzywuzzy**
- **Purpose:** String matching and fuzzy searching.
- **Usage in `card_reader_class.py`:**
  - Used for comparing OCR results to known key phrases and for autocorrecting OCR fields using similarity scores.
  - Example: `from fuzzywuzzy import fuzz, process`

---

## 2. **paddleocr==2.10.0**
- **Purpose:** Optical Character Recognition (OCR) engine.
- **Usage in `card_reader_class.py`:**
  - Used to extract text from card images.
  - Instantiated as `self.ocr_model = PaddleOCR(...)` and called in `card_reader` to perform OCR.

---

## 3. **paddlepaddle**
- **Purpose:** Deep learning framework required by PaddleOCR.
- **Usage in `card_reader_class.py`:**
  - Backend for PaddleOCR. Not directly imported, but required for OCR functionality.

---

## 4. **torch**
- **Purpose:** PyTorch, a deep learning framework.
- **Usage in `card_reader_class.py`:**
  - Used for loading and running custom dental marking detection models.
  - Example: `import torch`, `torch.load(...)`, `torch.manual_seed(...)`

---

## 5. **torchvision**
- **Purpose:** Image transformations and utilities for PyTorch.
- **Usage in `card_reader_class.py`:**
  - Used for preprocessing images before passing them to PyTorch models.
  - Example: `from torchvision import transforms`

---

## 6. **ultralytics**
- **Purpose:** Object detection models, especially YOLO.
- **Usage in `card_reader_class.py`:**
  - Used for detecting dental markings in images.
  - Example: `from ultralytics import YOLO`, `self.yolo_model = YOLO(...)`

---

## 7. **openpyxl**
- **Purpose:** Reading and writing Excel files.
- **Usage in `card_reader_class.py`:**
  - Used (commented out) for reading body part data from Excel files.
  - Example: `#self.body_part_df = pd.read_excel(...)`

---

## 8. **python-Levenshtein**
- **Purpose:** Fast string similarity calculations.
- **Usage in `card_reader_class.py`:**
  - Used as an optional speedup for fuzzywuzzy operations.

---

# Additional Standard Libraries Used

- **os, sys:** File and system operations.
- **cv2 (OpenCV):** Image processing and template matching.
- **numpy:** Numerical operations and array handling.
- **pandas:** Data manipulation and reading CSV/Excel files.
- **matplotlib, PIL:** Image display and annotation.

---

# Summary Table

| Package            | Main Function in card_reader_class.py                                  |
|--------------------|-----------------------------------------------------------------------|
| fuzzywuzzy         | Fuzzy string matching for OCR correction and key detection            |
| paddleocr          | OCR engine for extracting text from images                            |
| paddlepaddle       | Backend for PaddleOCR                                                |
| torch              | Deep learning inference for dental marking detection                  |
| torchvision        | Image preprocessing for PyTorch models                                |
| ultralytics        | YOLO object detection for dental marking localization                 |
| openpyxl           | (Optional) Excel file reading for body part data                      |
| python-Levenshtein | (Optional) Speedup for fuzzywuzzy string matching                     |

---