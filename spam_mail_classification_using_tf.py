import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer  #tokenizing
from tensorflow.keras.preprocessing.sequence import pad_sequences  #making the input into fixed length
import numpy as np

# Sample dataset
emails = [
    "Get free tickets now!",
    "Limited offer, buy now!",
    "Your wing is waiting",
    "Not spam",
    "Free trial, sign up now!"
]
labels = np.array([1, 1, 1, 0, 1], dtype=np.int32)  # 1 for spam, 0 for not spam

# Tokenize emails (use OOV token so unseen words map to a known index)
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(emails)
sequences = tokenizer.texts_to_sequences(emails)

# Vocabulary size for Embedding layer
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token index 0

# Pad sequences
max_length = 10
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=10, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary() #training the model

# Train model
model.fit(padded_sequences, labels, epochs=20, verbose=2)

# Test model
test_email = ["Get free stuff now!"]
test_sequence = tokenizer.texts_to_sequences(test_email)
test_padded_sequence = pad_sequences(test_sequence, maxlen=max_length, padding='post')
prediction = model.predict(test_padded_sequence)  # shape (1,1)

prob = float(prediction[0][0])
print(f"Spam probability: {prob:.4f}")
print("Spam" if prob > 0.5 else "Not Spam")
