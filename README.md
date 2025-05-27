# 🧠 Digit Classification from Scratch Using NumPy

This project implements a **fully connected feedforward neural network** from scratch using only **NumPy** to classify handwritten digits from the **MNIST** dataset (0–9).

---

## 📊 Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) consists of 28x28 grayscale images of handwritten digits.

- **Training data**: 60,000 images
- **Test data**: 10,000 images

---

## 🏗️ Neural Network Architecture

| Layer           | Type   | Size            | Activation |
|------------------|--------|------------------|------------|
| Input            | Dense | 784 (28×28)      | —          |
| Hidden Layer 1   | Dense | 128 neurons       | ReLU       |
| Hidden Layer 2   | Dense | 64 neurons        | ReLU       |
| Output           | Dense | 10 (classes 0–9)  | Softmax    |

---

## 🔧 Key Components

- **Activation Functions**:
  - `ReLU` (Rectified Linear Unit) is used in the hidden layers to introduce non-linearity.
  - `Softmax` is used in the output layer to convert raw output scores into probabilities.

- **Loss Function**:
  - `Cross-Entropy Loss` measures the difference between predicted and true label probabilities.

- **Optimization**:
  - Gradient Descent is manually implemented via **backpropagation**.
  - Weights and biases are updated after each training sample (stochastic training).

- **Initialization**:
  - Weights are initialized with small random values.
  - Biases are initialized to zero.

---

## 🧪 Model Training

- Each input image is:
  - Flattened from 28x28 to a 784-dimensional vector.
  - Normalized to values between 0 and 1.

- Training loop:
  - Forward pass → Loss computation → Backward pass (backpropagation) → Weight update
  - Training runs for a default of **20 epochs** (configurable).
  - Average loss per epoch is printed during training.

---

## ✅ Evaluation

- The model is tested on unseen data from the MNIST test set.
- The model outputs:
  - Probabilities for each digit (0–9).
  - Predicted digit class.
- Accuracy is calculated based on correct predictions out of all test samples.

---

## 🔍 Limitations

- Runs only on CPU (no GPU acceleration).
- No deep learning libraries (TensorFlow, PyTorch) are used.
- Trains one sample at a time (no batching or optimizers like Adam).
- Training on full dataset is slow compared to modern frameworks.

---

## 📌 Future Improvements

- [ ] Add **mini-batch training** for efficiency.
- [ ] Port to **TensorFlow** or **PyTorch** to support GPU training.

---

## 📁 Requirements

- Python 3.x
- NumPy
- Matplotlib
- TensorFlow (only used for loading the MNIST dataset)

```bash
pip install numpy matplotlib tensorflow
