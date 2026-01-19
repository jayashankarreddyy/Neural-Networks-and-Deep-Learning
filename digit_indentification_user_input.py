# mnist_predict_from_image.py
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "mnist_single_digit_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Saved model not found at {MODEL_PATH}. Please save your trained model as {MODEL_PATH} first.")

model = load_model(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

def preprocess_for_mnist(img_gray):
    if img_gray is None:
        raise ValueError("Input image is None")

    # ensure uint8
    img = img_gray.copy()
    if img.dtype != np.uint8:
        img = (255 * (img - img.min()) / (img.max() - img.min() + 1e-9)).astype(np.uint8)

    # Heuristic: check background polarity. MNIST digits are white(>0) on black(0) background.
    # If image has white background (mean > 127) and the digit is darker, invert.
    mean_val = img.mean()
    if mean_val > 127:
        img = 255 - img

    # Binary threshold to get cleaner region (use Otsu)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find content bounding box
    ys, xs = np.where(th > 0)
    if len(xs) > 0 and len(ys) > 0:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        # pad a bit
        pad = 4
        x1 = max(0, x1 - pad); x2 = min(img.shape[1]-1, x2 + pad)
        y1 = max(0, y1 - pad); y2 = min(img.shape[0]-1, y2 + pad)
        roi = th[y1:y2+1, x1:x2+1]
    else:
        # nothing detected, use the whole thresholded image
        roi = th

    # Resize roi to keep aspect ratio: target largest side ~20 pixels (as in MNIST)
    h, w = roi.shape
    if h == 0 or w == 0:
        # fallback to full image resized
        roi_resized = cv2.resize(th, (20, 20), interpolation=cv2.INTER_AREA)
        new_h, new_w = 20, 20
    else:
        scale = 20.0 / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place resized roi in 28x28 canvas centered
    canvas28 = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas28[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi_resized

    # Optional smoothing (Gaussian) to resemble MNIST stroke softness (kept minimal)
    canvas28 = cv2.GaussianBlur(canvas28, (3,3), 0)

    # Normalize to [0,1] as float32
    x = canvas28.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)  # shape (1,28,28)
    return x, canvas28

def predict_image_file(path, show_processed=False, save_processed_as=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    # Read as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    x, canvas28 = preprocess_for_mnist(img)
    preds = model.predict(x)  # shape (1,10)
    prob = float(np.max(preds, axis=1)[0])
    pred_label = int(np.argmax(preds, axis=1)[0])
    if show_processed:
        # Save or show processed 28x28 if requested (no GUI by default)
        if save_processed_as:
            cv2.imwrite(save_processed_as, canvas28)
    return pred_label, prob, canvas28

print("\nEnter the path to an image file (PNG/JPG). Type 'q' to quit.")
while True:
    user = input("Image path (or 'q' to quit): ").strip()
    if user.lower() == 'q':
        print("Exiting.")
        break
    if user == "":
        continue
    try:
        pred, conf, proc28 = predict_image_file(user, show_processed=True, save_processed_as="processed_28.png")
        print(f"Predicted digit: {pred}  (confidence: {conf:.3f})")
        print("Saved processed 28x28 image as 'processed_28.png' for inspection.")
    except Exception as e:
        print("Error:", e)
        print("Please provide a valid image path to a digit image (or type 'q' to quit).")
