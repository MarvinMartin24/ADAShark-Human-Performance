import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List

path = "data/"
random.seed(10)


def _input(message: str):
    """ Retreive user input only if correct
    Args:
        Message (str): Message displayed to user.
    """
    while True:
        try:
            res = input(message)
            if res in ["y", "n", "stop"]:
                if res == "stop":
                    return res
                return 1 if res == "y" else 0
            else:
                print("Wrong input, use y or n.")
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
    return (1 + np.square(beta)) / ((np.square(beta) / recall) + (1 /precision))

def metrics(preds: List, labels: List) -> str:
    """ Compute metrics including Accuracy, Precision, Recall, and F2 score.
    Args:
        precision (float)
        recall (float)
        beta (float)
    Returns:
        (str) "Accuracy: X%,Precision: X%, Recall: X%, F2 Score: X%",
    """
    d = {"TP":0, "TN":0, "FP":0,"FN":0}
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
        round(acc, 3) * 100 ,
        ",Precision: ",
        round(precision, 3) * 100,
        ",Recall: ",
         round(recall, 3) * 100,
        ",F2 Score: ",
        round(f2, 3) * 100
    )

if __name__ == '__main__':
    # Get Iamges
    images = []
    for i, vid in enumerate(os.listdir(path)):
        if (".png" in vid or ".jpg" in vid or ".jepg" in vid)and not "._" in vid:
            images.append(os.path.join(path, vid))

    # Select randomly 150 images (seed makes it repeatable)
    selected_images = random.sample(images, 150)

    preds = []
    human_labels = []
    labels = []

    for i, selected_image in enumerate(selected_images):
        img = Image.open(selected_image)
        img = np.asarray(img)
        plt.figure(figsize = (13,8))
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(img, interpolation='nearest')
        plt.draw()
        plt.pause(2) # pause how many seconds
        plt.close()
        labels.append(int(selected_image.split("_")[-3]))
        human_label = _input(f" {i}/{len(selected_images)} - Is there a shark? y/n/stop \n")
        if human_label != "stop":
            human_labels.append(int(human_label))
        else:
            break
        preds.append(int(selected_image.split("_")[-1].split(".")[0]))

    print("DACNN Metrics:", metrics(preds, labels))
    print("Human Metrics:", metrics(human_labels, labels))
