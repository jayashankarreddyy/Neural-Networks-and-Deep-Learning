#spices detection 
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image

# Load pre-trained model (ImageNet)
# First time it will download weights from the internet
model = MobileNetV2(weights="imagenet")


def classify_image_by_path(img_path: str):
    """
    Takes an image path, predicts the object,
    and groups it into: animal / bird / flower / other
    """
    if not os.path.exists(img_path):
        print(f"Path not found: {img_path}")
        return None

    # 1. Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # 2. Predict with the model
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=5)[0]  # list of (id, name, score)

    print("Top predictions:")
    for (_, name, score) in decoded:
        print(f"- {name:20s}  prob={score:.3f}")

    # 3. Decide high-level category: animal / bird / flower / other
    label_names = [name.lower() for (_, name, _) in decoded]

    # keywords for simple grouping
    bird_keywords = [
        "bird", "hen", "cock", "rooster", "ostrich", "flamingo",
        "partridge", "peacock", "sparrow", "parrot", "king_penguin", "goose"
    ]
    animal_keywords = [
        "dog", "cat", "puppy", "kitten", "horse", "cow", "bull", "sheep",
        "goat", "lion", "tiger", "elephant", "zebra", "bear", "monkey",
        "fox", "wolf", "leopard", "cheetah", "panda", "deer", "rabbit"
    ]
    flower_keywords = [
        "flower", "daisy", "sunflower", "tulip", "rose", "lotus",
        "orchid", "lily"
    ]

    def contains_any(label_list, keywords):
        return any(
            any(k in label for k in keywords)
            for label in label_list
        )

    if contains_any(label_names, bird_keywords):
        category = "bird"
    elif contains_any(label_names, animal_keywords):
        category = "animal"
    elif contains_any(label_names, flower_keywords):
        category = "flower"
    else:
        category = "other object"

    print(f"\nFinal category: {category}")
    return category, decoded


if name == "main":
    # Example: ask user for path
    path = input("Enter image path (e.g. C:/images/parrot.jpg): ")
    classify_image_by_path(path)
