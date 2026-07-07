"""
MNIST Handwritten Digit Classifier — Built From Scratch with NumPy
====================================================================

The neural network itself (forward prop, backprop, gradient descent) is
100% NumPy — no TensorFlow/PyTorch involved in the math. Keras is used
ONLY as a convenient way to fetch the raw MNIST arrays via:

    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

which returns:
    X_train: (60000, 28, 28) uint8   pixel values 0-255
    y_train: (60000,)        uint8   digit labels 0-9
    X_test:  (10000, 28, 28) uint8
    y_test:  (10000,)        uint8

Usage:
    python mnist_nn_scratch.py
    python mnist_nn_scratch.py --epochs 300 --lr 0.1 --hidden_size 128
"""

import argparse
import numpy as np


# ---------------------------------------------------------------------------
# 1. Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_mnist_from_keras():
    """
    Fetch MNIST via Keras' built-in loader (downloads/caches automatically
    the first time), then reshape it into the (features, examples) layout
    this network expects: (784, m) instead of (m, 28, 28).
    """
    from tensorflow import keras  # imported here so the rest of the file has zero TF/Keras dependency

    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    return prepare_data(X_train, y_train), prepare_data(X_test, y_test)


def prepare_data(X, y):
    """
    Convert (m, 28, 28) uint8 images and (m,) labels into the shapes
    the network works with:
      - X: (784, m) float32, normalized to [0, 1]
      - y: (m,) int labels, unchanged
    """
    m = X.shape[0]
    X_flat = X.reshape(m, 28 * 28).astype(np.float32) / 255.0  # (m, 784), normalize
    X_flat = X_flat.T  # (784, m) — each COLUMN is one example
    y = y.astype(int)
    return X_flat, y


def one_hot(labels, num_classes=10):
    """Convert integer labels (m,) into one-hot columns (num_classes, m)."""
    one_hot_y = np.zeros((labels.size, num_classes))
    one_hot_y[np.arange(labels.size), labels] = 1
    return one_hot_y.T


# ---------------------------------------------------------------------------
# 2. Parameter initialization
# ---------------------------------------------------------------------------
def init_params(input_size=784, hidden_size=128, output_size=10):
    """
    He initialization: scale weights by sqrt(2/fan_in). This keeps the
    variance of activations roughly constant across layers, which matters
    a lot when using ReLU (prevents vanishing/exploding activations).
    """
    W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2


# ---------------------------------------------------------------------------
# 3. Activation functions
# ---------------------------------------------------------------------------
def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(float)


def softmax(Z):
    # Subtract the max for numerical stability (avoids overflow in exp)
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# 4. Forward propagation
# ---------------------------------------------------------------------------
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# ---------------------------------------------------------------------------
# 5. Backward propagation
# ---------------------------------------------------------------------------
def backward_prop(Z1, A1, A2, W2, X, Y_one_hot, m):
    # Gradient of cross-entropy loss w.r.t. Z2 simplifies beautifully
    # when paired with softmax: dZ2 = A2 - Y
    dZ2 = A2 - Y_one_hot
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * relu_derivative(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


# ---------------------------------------------------------------------------
# 6. Parameter updates (vanilla gradient descent)
# ---------------------------------------------------------------------------
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2


# ---------------------------------------------------------------------------
# 7. Loss & accuracy
# ---------------------------------------------------------------------------
def compute_loss(A2, Y_one_hot):
    """Categorical cross-entropy loss, averaged over the batch."""
    m = Y_one_hot.shape[1]
    eps = 1e-8  # avoid log(0)
    return -np.sum(Y_one_hot * np.log(A2 + eps)) / m


def get_predictions(A2):
    return np.argmax(A2, axis=0)


def get_accuracy(predictions, labels):
    return np.sum(predictions == labels) / labels.size


# ---------------------------------------------------------------------------
# 8. Training loop — mini-batch gradient descent
# ---------------------------------------------------------------------------
def train(X, Y, hidden_size=128, epochs=200, lr=0.1, batch_size=64, verbose_every=25):
    input_size, m = X.shape
    output_size = int(Y.max()) + 1
    Y_one_hot_full = one_hot(Y, output_size)

    W1, b1, W2, b2 = init_params(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        # Shuffle each epoch for better generalization
        perm = np.random.permutation(m)
        X_shuffled = X[:, perm]
        Y_oh_shuffled = Y_one_hot_full[:, perm]

        for start in range(0, m, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_oh_shuffled[:, start:end]
            batch_m = X_batch.shape[1]

            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_batch)
            dW1, db1, dW2, db2 = backward_prop(
                Z1, A1, A2, W2, X_batch, Y_batch, batch_m
            )
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

        if epoch % verbose_every == 0 or epoch == epochs - 1:
            _, _, _, A2_full = forward_prop(W1, b1, W2, b2, X)
            loss = compute_loss(A2_full, Y_one_hot_full)
            acc = get_accuracy(get_predictions(A2_full), Y)
            print(f"Epoch {epoch:4d} | loss = {loss:.4f} | train acc = {acc:.4f}")

    return W1, b1, W2, b2


def predict(W1, b1, W2, b2, X):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)


# ---------------------------------------------------------------------------
# 9. Save / load trained weights (so you don't have to retrain every time)
# ---------------------------------------------------------------------------
def save_params(path, W1, b1, W2, b2):
    np.savez(path, W1=W1, b1=b1, W2=W2, b2=b2)


def load_params(path):
    data = np.load(path)
    return data["W1"], data["b1"], data["W2"], data["b2"]


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="From-scratch NumPy MNIST classifier")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dev_fraction", type=float, default=0.1,
                         help="Fraction of the training set held out for validation")
    parser.add_argument("--save_path", default="mnist_weights.npz")
    args = parser.parse_args()

    print("Loading MNIST via keras.datasets.mnist.load_data()...")
    (X_train_full, Y_train_full), (X_test, Y_test) = load_mnist_from_keras()

    m_total = X_train_full.shape[1]
    dev_size = int(m_total * args.dev_fraction)

    # Hold out part of the training set as a dev/validation split
    X_dev, Y_dev = X_train_full[:, :dev_size], Y_train_full[:dev_size]
    X_train, Y_train = X_train_full[:, dev_size:], Y_train_full[dev_size:]

    print(f"Training on {X_train.shape[1]} examples, "
          f"validating on {X_dev.shape[1]} examples, "
          f"final test set has {X_test.shape[1]} examples")

    W1, b1, W2, b2 = train(
        X_train, Y_train,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    dev_preds = predict(W1, b1, W2, b2, X_dev)
    print(f"\nDev accuracy: {get_accuracy(dev_preds, Y_dev):.4f}")

    test_preds = predict(W1, b1, W2, b2, X_test)
    print(f"Test accuracy: {get_accuracy(test_preds, Y_test):.4f}")

    save_params(args.save_path, W1, b1, W2, b2)
    print(f"Saved trained weights to {args.save_path}")


if __name__ == "__main__":
    main()
