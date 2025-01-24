import numpy as np


class Activations:
    @staticmethod
    def get(name):
        match name.lower():
            case "linear":
                return Activations.linear, Activations.linear_derivative
            case "sigmoid":
                return Activations.sigmoid, Activations.sigmoid_derivative
            case "tanh":
                return Activations.tanh, Activations.tanh_derivative
            case "relu":
                return Activations.relu, Activations.relu_derivative
            case "softmax":
                return Activations.softmax, Activations.softmax_derivative
            case _:
                return Activations.relu, Activations.relu_derivative

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return Activations.sigmoid(x) * (1.0 - Activations.sigmoid(x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        value = Activations.tanh(x)
        return 1.0 - value * value

    @staticmethod
    def relu(x):
        return x * (x > 0)

    @staticmethod
    def relu_derivative(x):
        return 1.0 * (x > 0)

    @staticmethod
    def softmax(x):
        x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

    @staticmethod
    def softmax_derivative(x):
        softmax_output = Activations.softmax(x)

        num_samples = softmax_output.shape[0]
        num_classes = softmax_output.shape[1]

        jacobian_matrix = np.zeros((num_samples, num_classes, num_classes))

        for i in range(num_samples):
            for j in range(num_classes):
                for k in range(num_classes):
                    if j == k:
                        jacobian_matrix[i, j, k] = softmax_output[i, j] * (
                            1 - softmax_output[i, j]
                        )
                    else:
                        jacobian_matrix[i, j, k] = (
                            -softmax_output[i, j] * softmax_output[i, k]
                        )

        return jacobian_matrix
