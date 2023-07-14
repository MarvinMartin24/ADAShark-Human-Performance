import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
import fire

path = "data/"

def _input(message: str):
    """ Retrieve user input only if correct
    Args:
        message (str): Message displayed to the user.
    """
    while True:
        try:
            res = input(message)
            if res in ["y", "n", "stop"]:
                if res == "stop":
                    return res
                return 1 if res == "y" else 0
            else:
                print("Wrong input, use y or n or stop.")
        except:
            pass

def fbeta_score(precision: float, recall: float, beta: float) -> float:
    """ Compute Fbeta score.
    Args:
        precision (float)
        recall (float)
        beta (float)
    Returns:
        (float) Fbeta score.
    """
    return (1 + np.square(beta)) / ((np.square(beta) / recall) + (1 / precision))

def metrics(preds: List, labels: List) -> str:
    """ Compute metrics including Accuracy, Precision, Recall, and F2 score.
    Args:
        precision (float)
        recall (float)
        beta (float)
    Returns:
        (str) "Accuracy: X%, Precision: X%, Recall: X%, F2 Score: X%",
    """
    d = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for pred, label in zip(preds, labels):
        if (pred, label) == (1, 1):
            d['TP'] += 1
        if (pred, label) == (0, 0):
            d['TN'] += 1
        if (pred, label) == (1, 0):
            d['FP'] += 1
        if (pred, label) == (0, 1):
            d['FN'] += 1

    precision = d['TP'] / (d['TP'] + d['FP'] + 0.0001)
    recall = d['TP'] / (d['TP'] + d['FN'] + 0.0001)
    acc = (d['TP'] + d['TN']) / (len(preds) + 0.0001)
    f2 = fbeta_score(precision, recall, beta=2)

    return (
        "Accuracy: ",
        round(acc, 3) * 100,
        ", Precision: ",
        round(precision, 3) * 100,
        ", Recall: ",
        round(recall, 3) * 100,
        ", F2 Score: ",
        round(f2, 3) * 100
    )

def main(assistance: bool = False):
    # Get images
    images = []
    for i, vid in enumerate(os.listdir(path)):
        if (".png" in vid or ".jpg" in vid or ".jpeg" in vid) and not "._" in vid:
            images.append(os.path.join(path, vid))

    if assistance:
        random.seed(99)
    else:
        random.seed(10)

    # Select randomly 150 images (seed makes it repeatable)
    selected_images = random.sample(images, 150)

    preds = []
    human_labels = []
    labels = []
    traces = []

    for i, selected_image in enumerate(selected_images):
        prob = selected_image.split('_')[-2]

        if assistance:
            # Display black frame for 2 seconds
            plt.figure(figsize=(13, 8))
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
            plt.text(50, 50, f"Shark Probabilty: {prob}%", fontsize=24, color='white', ha='center', va='center')
            plt.draw()
            plt.pause(2)
            plt.close()

        img = Image.open(selected_image)
        img = np.asarray(img)
        plt.figure(figsize=(13, 8))
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(img, interpolation='nearest')
        if assistance:
            plt.text(10, 20, f"Prob: {prob}%", fontsize=12, color='white', ha='left', va='top')
        plt.draw()
        plt.pause(2)
        plt.close()

        label = int(selected_image.split("_")[-3])
        labels.append(label)
        if assistance:
            message = f" {i + 1}/{len(selected_images)} - Is there a shark? y/n/stop ({prob}%)\n"
            output_file = "round_2_metrics.txt"
        else:
            message = f" {i + 1}/{len(selected_images)} - Is there a shark? y/n/stop \n"
            output_file = "round_1_metrics.txt"

        human_label = _input(message)
        if human_label != "stop":
            human_labels.append(int(human_label))
        else:
            break
        pred_bin = int(selected_image.split("_")[-1].split(".")[0])
        preds.append(pred_bin)
        traces.append({
            "img": selected_image,
            "label": label,
            "human": human_label,
            "DACNN": (prob, pred_bin)
        })

    with open(output_file, 'w') as f:
        f.write(f"DACNN Metrics: {metrics(preds, labels)}\nHuman Metrics {metrics(human_labels, labels)}\n")
        f.write(f"Trace: {traces}")


if __name__ == '__main__':
    fire.Fire(main)
