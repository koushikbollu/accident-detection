# Automatic Accident Detection Using Deep Learning

A deep-learning system that detects road accidents from images and sends an instant alert, with the accident photo, to Telegram.

## What it does

1. Prepares a cleaned accident dataset (train / valid / test splits, 2 classes: accident and non-accident) — renaming images, generating YOLO labels and resizing.
2. Trains a YOLOv8 detection model on the dataset (`data.yaml`, training and validation included).
3. Runs detection with a vehicle-detection model and collision-detection logic to decide whether an accident has occurred.
4. On detection, sends an alert message and the accident image to Telegram using the Telegram Bot API.

## Tech stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch / torchvision
- Telegram Bot API (requests)
- Google Colab

## Results

- Accuracy: 98.28%
- Precision: 97.15%
- Recall: 99.03%

Sample outputs are in `output_photos.pdf`.

## Files

- `accident_detection.py` — the complete project code, section by section: dataset preparation, label generation, `data.yaml` creation, YOLOv8 training and validation, Telegram alert functions, and the final detection system.
- `requirements.txt` — libraries to install.
- `output_photos.pdf` — sample detection outputs.

## How to run

1. Open `accident_detection.py` in Google Colab (it mounts Google Drive for the dataset).
2. Install the libraries in `requirements.txt`.
3. Point `base_path` in the code to your dataset folder (train/valid/test, images and labels).
4. Set your own Telegram `BOT_TOKEN` and `CHAT_ID` in the Telegram configuration section.
5. Run the sections in order: dataset prep → training → validation → final detection system.

## Note

The cleaned accident dataset (about 1.3 GB) and the trained model weights are not included in this repository because of GitHub file-size limits.
