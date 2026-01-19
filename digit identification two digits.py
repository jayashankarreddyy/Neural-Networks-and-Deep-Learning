import numpy as np
import struct
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

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

print("Loaded shapes (single-digit):")
print(" train_images:", train_images.shape, " train_labels:", train_labels.shape)
print(" test_images: ", test_images.shape, " test_labels: ", test_labels.shape)

# 2. Make paired images (left+right) by concatenating horizontally
def make_pairs(images, labels, pair_count=None, rng_seed=1):
    """
    Create pairs by randomly sampling two indices (with replacement).
    Returns X shape (N, 28, 56), y_left (N,), y_right (N,)
    """
    n = len(images)
    if pair_count is None:
        pair_count = n

    rng = np.random.default_rng(rng_seed)
    idx1 = rng.integers(0, n, size=pair_count)
    idx2 = rng.integers(0, n, size=pair_count)

    X = np.empty((pair_count, 28, 56), dtype=np.float32)
    yL = np.empty((pair_count,), dtype=np.int32)
    yR = np.empty((pair_count,), dtype=np.int32)

    for i, (a, b) in enumerate(zip(idx1, idx2)):
        left = images[a]
        right = images[b]
        X[i] = np.concatenate([left, right], axis=1)  # horizontal concat -> width 56
        yL[i] = labels[a]
        yR[i] = labels[b]

    return X, yL, yR

PAIR_COUNT_TRAIN = 60000  # you can reduce for faster experiments
PAIR_COUNT_TEST  = 10000

X_train_all, yL_train_all, yR_train_all = make_pairs(train_images, train_labels, pair_count=PAIR_COUNT_TRAIN, rng_seed=1)
X_test, yL_test, yR_test = make_pairs(test_images, test_labels, pair_count=PAIR_COUNT_TEST, rng_seed=2)

# Normalize to [0,1] and add channel dimension if desired
X_train_all = (X_train_all / 255.0).astype(np.float32)
X_test = (X_test / 255.0).astype(np.float32)

# Optional: split a validation set from training
X_train, X_val, yL_train, yL_val, yR_train, yR_val = train_test_split(
    X_train_all, yL_train_all, yR_train_all, test_size=0.1, random_state=42
)

print("Paired shapes:")
print(" X_train:", X_train.shape, " yL_train:", yL_train.shape, " yR_train:", yR_train.shape)
print(" X_val:  ", X_val.shape)
print(" X_test: ", X_test.shape)

# Convert labels to categorical (one-hot)
yL_train_cat = to_categorical(yL_train, 10)
yR_train_cat = to_categorical(yR_train, 10)
yL_val_cat   = to_categorical(yL_val, 10)
yR_val_cat   = to_categorical(yR_val, 10)
yL_test_cat  = to_categorical(yL_test, 10)
yR_test_cat  = to_categorical(yR_test, 10)

''' 3. Build a model with two output heads (left and right digit)
    -- using a simple flattened Dense backbone to keep similarity with your original model'''
inp = Input(shape=(28, 56), name="input_pair")   # no explicit channel dim
x = Flatten()(inp)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)

# left head
left = Dense(64, activation='relu')(x)
left_out = Dense(10, activation='softmax', name='left_digit')(left)

# right head
right = Dense(64, activation='relu')(x)
right_out = Dense(10, activation='softmax', name='right_digit')(right)

model = Model(inputs=inp, outputs=[left_out, right_out])

model.compile(optimizer='adam',
              loss={'left_digit': 'categorical_crossentropy', 'right_digit': 'categorical_crossentropy'},
              metrics={'left_digit': 'accuracy', 'right_digit': 'accuracy'})

model.summary()

# 4. Train (both heads)
EPOCHS = 5
BATCH_SIZE = 32

history = model.fit(
    X_train, {'left_digit': yL_train_cat, 'right_digit': yR_train_cat},
    validation_data=(X_val, {'left_digit': yL_val_cat, 'right_digit': yR_val_cat}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# 5. Evaluate on test pairs
eval_res = model.evaluate(X_test, {'left_digit': yL_test_cat, 'right_digit': yR_test_cat}, verbose=2)
print("\nTest eval (total_loss, left_loss, right_loss, left_acc, right_acc):", eval_res)

# 6. Quick predictions preview
n = 8
preds_left, preds_right = model.predict(X_test[:n])
pred_left_labels = np.argmax(preds_left, axis=1)
pred_right_labels = np.argmax(preds_right, axis=1)

print("\nExample (ground_truth_left, ground_truth_right) -> (pred_left, pred_right):")
for i in range(n):
    print((int(yL_test[i]), int(yR_test[i])), "->", (int(pred_left_labels[i]), int(pred_right_labels[i])))
