# ADAShark-Human-Performance
Scripts to compare ADAShark vs Human Performance.

## Introduction
Labels, ADAShark inference confidences/predictions are hardcoded in the filename `XXX_label_confidence_prediction.format`.

## Usage
```
pip install -r requirements.txt
python3 main.py
```
First look at the image for 2 seconds and give the label when the image disappears.
- `y` means one or multiple sharks are present in the image.
- `n` means no shark is present in the image.
- `stop` will stop the labeling process and return the metrics obtained so far.

## Output
Expected output format in Percentages:
```
DACNN Metrics: ('Accuracy: ', 76.0, ',Precision: ', 65.10000000000001, ',Recall: ', 90.3, ',F2 Score: ', 83.8)
Human Metrics: ('Accuracy: ', 92.7, ',Precision: ', 93.2, ',Recall: ', 88.7, ',F2 Score: ', 89.60000000000001)
```
