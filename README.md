# 🧠 Building a Digit Classification Model from scratch Using NumPy

This project implements a **fully connected feedforward neural network** from scratch using only **NumPy** to classify handwritten digits from the **MNIST** dataset (0–9).

---

## 📊 Dataset

The MNIST dataset consists of 28x28 grayscale images of handwritten digits.

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

---

## 🔍 Limitations

- Runs only on CPU (no GPU acceleration).
- No deep learning libraries (TensorFlow, PyTorch) are used.
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




## 🧠 How the Model Works (Simplified Explanation)

The model consists of **4 layers** in total:
- **1 Input Layer**
- **2 Hidden (Intermediate) Layers**
- **1 Output Layer**

The **input layer** takes a 1D array representing the image — in the case of MNIST, each image is 28×28 pixels, which is flattened into a vector of **784 values**.

There can be **any number of hidden layers**, and each can have **any number of neurons**. In our case:
- The first hidden layer has **128 neurons**
- The second hidden layer has **64 neurons**

The data flows through the network as follows:
1. The input vector is multiplied with the weights of the first hidden layer and passed through an activation function (ReLU).
2. The result becomes the input to the second hidden layer, and the same process repeats.
3. Finally, the output from the last hidden layer is passed to the **output layer**, which has **10 neurons** — each representing one digit (0 through 9).

The output is a **probability distribution** over the 10 digits, calculated using the **softmax function**.  
The digit with the highest probability is chosen as the predicted output.

---

To evaluate how well the model is performing, we use a **loss function** (cross-entropy loss), which measures the difference between the predicted output and the actual label.  
The **goal of training** is to minimize this loss by adjusting the **weights and biases** in the network.

This process is repeated using many training examples, which helps the model learn better and generalize to unseen data.

> 📌 **Note**: The more training data the model sees, the better it can adjust its internal parameters to make accurate predictions.

For example, training the model on just 10 samples wouldn't be effective, because it wouldn’t see enough variation to learn meaningful patterns.  
A larger dataset helps the model **generalize better** and **reduces the loss** over time.

```bash
pip install numpy matplotlib tensorflow
