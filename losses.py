import numpy as np
from activations import Activations


class Losses:
    @staticmethod
    def get(name):
        match name.lower():
            case "mean_squared_error":
                return Losses.mean_squared_error, Losses.mean_squared_error_derivative
            case "binary_cross_entropy":
                return (
                    Losses.binary_cross_entropy,
                    Losses.binary_cross_entropy_derivative,
                )
            case "categorical_cross_entropy":
                return (
                    Losses.categorical_cross_entropy,
                    Losses.categorical_cross_entropy_derivative,
                )
            case _:
                return Losses.mean_squared_error, Losses.mean_squared_error_derivative

    @staticmethod
    def simplify(loss, output_activation):
        loss = loss.lower()
        output_activation = output_activation.lower()

        if loss == "binary_cross_entropy" and output_activation == "sigmoid":
            return Losses.mean_squared_error_derivative, Activations.linear_derivative

        if loss == "categorical_cross_entropy" and output_activation == "softmax":
            return Losses.mean_squared_error_derivative, Activations.linear_derivative

        return Losses.get(loss)[1], Activations.get(output_activation)[1]

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        error = y_pred - y_true
        return np.mean(error * error)

    @staticmethod
    def mean_squared_error_derivative(y_true, y_pred):
        return y_pred - y_true

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred))

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return np.mean(-np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def categorical_cross_entropy_derivative(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -y_true / y_pred
