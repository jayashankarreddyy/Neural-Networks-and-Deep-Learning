import numpy as np
import struct
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


# 1. Load IDX Image and Label Files
def load_images(filepath):
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
        return images

def load_labels(filepath):
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

train_images = load_images(r"C:\Users\Suwin\Desktop\Deep Learning\train-images.idx3-ubyte")
train_labels = load_labels(r"C:\Users\Suwin\Desktop\Deep Learning\train-labels.idx1-ubyte")

test_images = load_images(r"C:\Users\Suwin\Desktop\Deep Learning\t10k-images.idx3-ubyte")
test_labels = load_labels(r"C:\Users\Suwin\Desktop\Deep Learning\t10k-labels.idx1-ubyte")

print("Loaded shapes:")
print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

# 2. Normalize + Prepare Labels
train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels_cat = to_categorical(train_labels, 10)
test_labels_cat = to_categorical(test_labels, 10)


# 3. Build a Simple Neural Network
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4. Train the Model
model.fit(train_images, train_labels_cat, epochs=5, batch_size=32)

# 5. Evaluate
loss, acc = model.evaluate(test_images, test_labels_cat)
print(f"\nTest Accuracy: {acc:.4f}")
