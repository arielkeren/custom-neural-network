# Custom Neural Network

A custom implementation of a neural network only with Python and Numpy.

## Features

### Components

`Input` - Defines the number of input neurons as a parameter.

#### Dense (Fully-Connected) Layers

All of these get the number of neurons as a parameter.

- `Linear` - Dense layer with no activation.
- `Relu` - Dense layer with ReLU (Rectified Linear Unit) activation.
- `Sigmoid` - Dense layer with sigmoid activation.
- `Tanh` - Dense layer with tanh (hyperbolic tangent) activation.
- `Softmax` - Dense layer with softmax activation.

#### Losses

- `MeanSquaredError` - Mean squared error loss.
- `BinaryCrossEntropy` - Binary cross entropy loss.
- `CategoricalCrossEntropy` - Categorical cross entropy loss.
- `SparseCategoricalCrossEntropy` - Sparse categorical cross entropy loss.

### Model Creation

A new model can be created by using the `+` operator (addition) with components or already-defined models.<br>
For example, a simple binary classifier can be created as such:<br>
```py
model = Input(2) + Relu(32) + Relu(16) + Sigmoid(1) + BinaryCrossEntropy()
```
It can also be broken into pieces as such:
```py
input_layer = Input(2)
hidden_layers = Relu(32) + Relu(16)
output_layer = Sigmoid(1) + BinaryCrossEntropy()
model = input_layer + hidden_layers + output_layer
```

### Model Functions

- `summary` - Prints all the information about the model (layers, loss function, total parameters, etc.)
- `evaluate` - Tests the model on the given test data, and prints the accuracy.
- `predict` - Runs the model on the given input values, and returns the model's output.
- `fit` - Trains the model, given the training data, number of epochs, learning rate, batch size and validation data (optional). Returns the loss history and the validation loss history.

### Utility Functions

- `split_data` - Splits the given `x` and `y` arrays into `x_train`, `y_train`, `x_test` and `y_test`, based on the given test percentage.
- `one_hot_encode` - Encodes the given array into its one hot representation (to use **categorical cross entropy**).
- `one_hot_decode` - Decodes the given array from its one hot representation (to use **sparse categorical cross entropy**).
- `plot_loss` - Plots the given loss history and the given validation loss history (optional) with **Matplotlib**.

## Example

A simple neural network to predict the XOR of the inputs:
```py
import numpy as np
from components import *
from utils import Utils

if __name__ == "__main__":
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    model = Input(2) + Relu(32) + Relu(16) + Softmax(2) + SparseCategoricalCrossEntropy()

    history = model.fit(x, y, epochs=100, learning_rate=0.1, batch_size=4)
    model.evaluate(x, y)
    model.summary()

    Utils.plot_loss(history["loss"])
```
The output in the terminal:
```
Accuracy: 100.0%
----------
Loss: sparse_categorical_cross_entropy
Input Layer: 2

--- Hidden Layer 1 ---
Neurons: 32
Activation: relu
Initialization: he
Weights: 64
Biases: 32

--- Hidden Layer 2 ---
Neurons: 16
Activation: relu
Initialization: he
Weights: 512
Biases: 16

--- Output Layer ---
Neurons: 2
Activation: softmax
Initialization: xavier
Weights: 32
Biases: 2

Total weights: 608
Total biases: 50
Total parameters: 658
----------
```

## Instructions

- Clone this repository:
   ```bash
   git clone https://github.com/arielkeren/custom-neural-network.git
   ```
- Create a new Python file.
- Import the components and utility functions:
   ```py
   from components import *
   from utils import Utils
   ```
