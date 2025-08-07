# Backpropagation Neural Network from Scratch

This repository demonstrates a **simple neural network** built entirely **from scratch using NumPy**, without any ML libraries like TensorFlow or PyTorch. It predicts **LPA (Salary in Lakhs Per Annum)** from **CGPA** and **Profile Score** using a **2-layer neural network** with manual weight updates.

---

## 📌 Objective

To understand how a neural network works internally by:
- Initializing weights manually
- Performing forward propagation without activations
- Manually updating weights using a basic loss derivative

---

## 📈 Dataset

A small synthetic dataset with the following features:
- `cgpa` (Cumulative Grade Point Average)
- `profile_score` (A custom score based on profile strength)
- `lpa` (Target: Salary in LPA)

Example:

| cgpa | profile_score | lpa |
|------|----------------|-----|
| 8    | 8              | 4   |
| 7    | 9              | 5   |
| 6    | 10             | 6   |
| 5    | 12             | 7   |

---

## 🧠 Neural Network Architecture

- **Input Layer:** 2 features (`cgpa`, `profile_score`)
- **Hidden Layer:** 2 neurons (no activation function)
- **Output Layer:** 1 neuron (predicts `lpa`)

The forward pass:

```text
Input (2D) → [W1, b1] → Hidden (2D) → [W2, b2] → Output (1D)
```

No activation functions are used for simplicity. Loss is calculated as Mean Squared Error (MSE).

---

## 🔁 Training Procedure

- Number of epochs: 5 (can be changed)
- Loss function: MSE (Mean Squared Error)
- Manual update rule using gradient approximations
- Learning rate: hardcoded in update formulas (`0.001`)

---

## 🧾 Example Code Snippet

```python
# Forward Propagation
def L_layer_forward(X, parameters):
    A = X
    L = len(parameters) // 2
    for l in range(1, L+1):
        A_prev = A
        Wl = parameters['W' + str(l)]
        bl = parameters['b' + str(l)]
        A = np.dot(Wl.T, A_prev) + bl
    return A, A_prev

# Update Parameters (Manual)
def update_parameters(parameters, y, y_hat, A1, X):
    # Output layer gradients
    error = 0.001 * 2 * (y - y_hat)
    parameters['W2'][0][0] += error * A1[0][0]
    parameters['W2'][1][0] += error * A1[1][0]
    parameters['b2'][0][0] += error

    # Hidden layer gradients
    parameters['W1'][0][0] += error * parameters['W2'][0][0] * X[0][0]
    parameters['W1'][0][1] += error * parameters['W2'][0][0] * X[1][0]
    parameters['b1'][0][0] += error * parameters['W2'][0][0]

    parameters['W1'][1][0] += error * parameters['W2'][1][0] * X[0][0]
    parameters['W1'][1][1] += error * parameters['W2'][1][0] * X[1][0]
    parameters['b1'][1][0] += error * parameters['W2'][1][0]
```

---

## 🧪 Training Output

During training, average loss per epoch is printed. Sample:

```
Epoch -  1 Loss -  25.321744156025517
Epoch -  2 Loss -  18.320004165722047
Epoch -  3 Loss -  9.473661050729628
Epoch -  4 Loss -  3.2520938634031613
Epoch -  5 Loss -  1.3407132589299962
```

## 🚀 Usage

1. Clone the repo:

```bash
git clone https://github.com/your-username/backpropagation-nn-from-scratch.git
cd backpropagation-nn-from-scratch
```

2. Run the Python script or Jupyter Notebook:

```bash
python train.py
# OR
jupyter notebook backpropagation-regression.ipynb
```

---

## 📚 Learning Goals

✅ Understand data flow through layers  
✅ Learn how weights and biases are updated  
✅ Grasp forward propagation without external libraries  
✅ Build intuition for gradient-based optimization

---

