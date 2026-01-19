import numpy as np

# activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def classify_email(x):
    # --- Given weights from your notes ---
    # Two hidden neurons (each has 3 input weights)
    W_hidden = np.array([
        [0.5, -0.2,  0.3],   # weights for hidden neuron 1
        [0.4,  0.1, -0.5]    # weights for hidden neuron 2
    ])  # shape (2,3)

    # Hidden -> output weights (one weight per hidden neuron)
    W_out = np.array([0.7, 0.2])  # shape (2,)

    # --- Step 1: Weighted sums for hidden neurons ---
    z_hidden = W_hidden.dot(x)    # shape (2,)
    print("Weighted sums (z_hidden):", z_hidden)  # expect [0.8, -0.1]

    # --- Step 2: Activation for hidden layer (ReLU) ---
    h = relu(z_hidden)
    print("Hidden activations after ReLU (h):", h)  # expect [0.8, 0.0]

    # --- Step 3: Weighted sum for output ---
    z_out = np.dot(W_out, h)      # scalar; (0.8*0.7 + 0*0.2) = 0.56
    print("Weighted sum at output (z_out):", z_out)

    # --- Step 4: Activation for output (Sigmoid) ---
    prob = sigmoid(z_out)         # scalar, ~0.636
    print("Output after Sigmoid (probability):", prob)

    # --- Step 5: Human-readable message + decision ---
    prob_percent = prob * 100
    message = f"The mail is {prob_percent:.2f}% likely to be spam."
    label = "SPAM" if prob > 0.5 else "NOT SPAM"

    return prob, message, label

# Example run using your input vector [1, 0, 1]
x = np.array([1, 0, 1])
prob, message, label = classify_email(x)

print()
print(message)
print("Decision (threshold 0.5):", label)
