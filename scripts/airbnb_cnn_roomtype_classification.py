# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3.10 (tfenv)
#     language: python
#     name: tfenv
# ---

# %% id="XgxqfHiw9O1X"
import os
import ssl
import requests
from io import BytesIO
from urllib import request

from tqdm import tqdm
from PIL import Image

import pandas as pd
import json
import numpy as np
import seaborn as sn
import pickle

from matplotlib import pyplot as plt
from io import StringIO
# %matplotlib inline

# %% id="3bzuJKLx-D2s"
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_datasets as tfds

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# %% colab={"base_uri": "https://localhost:8080/"} id="KWhTjjVNb7Zi" outputId="6720e9af-4d42-440d-902b-b839a6bd9553"
import cv2

# %%
from vgg16_places_365 import VGG16_Places365
model = VGG16_Places365(weights=None)
model.summary()

# %%
model = VGG16_Places365(weights='places')

with open('categories_places365.txt') as f:
    classes = [line.strip().split()[0][1:] for line in f.readlines()]

print("Model loaded successfully! Ready for kitchen/dining_room inference.")
print(f"Total classes: {len(classes)}")
print(f"Contains 'kitchen': {'kitchen' in classes}")
print(f"Contains 'dining_room': {'dining_room' in classes}")


# %%
def predict_scene(image_path):
    # Preprocess (same as before)
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, 0)
    
    # Predict
    preds = model.predict(img)[0]
    top_idx = np.argsort(preds)[-5:][::-1]
    
    print("Top 5 predictions:")
    for i, idx in enumerate(top_idx):
        print(f"{i+1}. {classes[idx]}: {preds[idx]:.1%}")
    
    # Kitchen vs Dining room
    kitchen_idx = classes.index('k/kitchen')
    dining_idx = classes.index('d/dining_room')
    print(f"\nKitchen: {preds[kitchen_idx]:.1%}")
    print(f"Dining room: {preds[dining_idx]:.1%}")
    print("Result:", "Dining room" if preds[dining_idx] > preds[kitchen_idx] else "Kitchen")


# %%
predict_scene('kitchen.png')

# %%
