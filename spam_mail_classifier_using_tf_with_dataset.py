
# train_spam_tf.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

DATA_PATH = "spam_mail_classifier.csv"  # path to your uploaded CSV

# 1) Load CSV
df = pd.read_csv(r"C:\Users\Suwin\Desktop\Deep Learning\spam_mail_classifier.csv")
print("Loaded dataset shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head(5))

# 2) Detect text and label columns (best-effort)
# You can override these if your CSV uses different names:
# text_col = "message"
# label_col = "label"

# Auto-detect:
obj_cols = [c for c in df.columns if df[c].dtype == "object"]
text_col = obj_cols[0] if obj_cols else df.columns[0]
label_col = None
for c in df.columns:
    if c != text_col and df[c].nunique() <= 10:
        label_col = c
        break
if label_col is None:
    label_col = df.columns[-1]

print(f"Using text column: '{text_col}', label column: '{label_col}'")

texts = df[text_col].astype(str).fillna("")
labels_raw = df[label_col].astype(str).fillna("")

# 3) Map labels to binary 0/1 (common heuristics)
def map_label(v):
    vs = v.lower().strip()
    if vs in ("spam","1","true","t","yes","y","s"):
        return 1
    if vs in ("ham","not spam","0","false","f","no","n"):
        return 0
    # fallback: contains 'spam'
    return 1 if "spam" in vs else 0

try:
    # If numeric 0/1, use numeric mapping
    num = pd.to_numeric(labels_raw)
    unique_nums = sorted(num.unique())
    if len(unique_nums) == 2 and set(unique_nums) <= {0,1}:
        labels = num.astype(int)
    else:
        labels = labels_raw.map(map_label).astype(int)
except Exception:
    labels = labels_raw.map(map_label).astype(int)

print("Label counts after mapping:\n", pd.Series(labels).value_counts())

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print("Train / Test sizes:", len(X_train), len(X_test))

# 5) Tokenize + pad
NUM_WORDS = 10000
MAX_LEN = 100
tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

train_seq = tokenizer.texts_to_sequences(X_train)
test_seq = tokenizer.texts_to_sequences(X_test)

padded_train = pad_sequences(train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
padded_test = pad_sequences(test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# 6) Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=NUM_WORDS, output_dim=32, input_length=MAX_LEN),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 7) Train
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    padded_train, np.array(y_train),
    validation_split=0.1,
    epochs=20,
    batch_size=32,
    callbacks=[es],
    verbose=2
)

# 8) Evaluate
pred_probs = model.predict(padded_test).ravel()
preds = (pred_probs >= 0.5).astype(int)

print("Test accuracy:", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds, zero_division=0))
print("Recall:", recall_score(y_test, preds, zero_division=0))
print("F1:", f1_score(y_test, preds, zero_division=0))
print("\nClassification report:\n", classification_report(y_test, preds, zero_division=0))

# 9) Save model & tokenizer
os.makedirs("artifacts", exist_ok=True)
model.save("artifacts/spam_text_model.h5")
with open("artifacts/spam_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("Saved model and tokenizer to ./artifacts/")
