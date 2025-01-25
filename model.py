import numpy as np
import matplotlib.pyplot as plt
from components import Input, Loss
from layer import Layer
from losses import Losses
from activations import Activations
from initializations import Initializations


class Model:
    def __init__(self, num_inputs, loss):
        self.num_inputs = num_inputs
        self.loss_name = loss
        self.loss, self.loss_derivative = Losses.get(loss)
        self.weights = []
        self.biases = []
        self.activations = []
        self.derivatives = []
        self.layers = []
        self.error_history = []

    def add(self, layer):
        if len(self.layers) == 0:
            self.weights.append(
                Initializations.get(layer.initialization)(
                    self.num_inputs, layer.num_neurons
                )
            )
        else:
            self.weights.append(
                Initializations.get(layer.initialization)(
                    self.weights[-1].shape[1], layer.num_neurons
                )
            )

        self.layers.append(
            Layer(layer.num_neurons, layer.activation, layer.initialization)
        )
        self.biases.append(np.zeros(layer.num_neurons))

        activation_function, derivative = Activations.get(layer.activation)
        self.activations.append(activation_function)
        self.derivatives.append(derivative)

    def remove(self):
        if len(self.layers) == 0:
            return

        self.weights = self.weights[:-1]
        self.biases = self.biases[:-1]
        self.activations = self.activations[:-1]
        self.derivatives = self.derivatives[:-1]
        self.layers = self.layers[:-1]

    def summary(self):
        print("----------")

        print("Loss:", self.loss_name)
        print("Input Layer:", self.num_inputs)

        total_weights = 0
        total_biases = 0
        last_num_neurons = self.num_inputs

        for index, layer in enumerate(self.layers[:-1]):
            total_weights += layer.num_neurons * last_num_neurons
            total_biases += layer.num_neurons

            print("\n--- Hidden Layer", index + 1, "---")
            print("Neurons:", layer.num_neurons)
            print("Activation:", layer.activation)
            print("Initialization:", layer.initialization)
            print("Weights:", layer.num_neurons * last_num_neurons)
            print("Biases:", layer.num_neurons)

            last_num_neurons = layer.num_neurons

        output_layer = self.layers[-1]
        total_weights += output_layer.num_neurons * last_num_neurons
        total_biases += output_layer.num_neurons

        print("\n--- Output Layer ---")
        print("Neurons:", output_layer.num_neurons)
        print("Activation:", output_layer.activation)
        print("Initialization:", output_layer.initialization)
        print("Weights:", output_layer.num_neurons * last_num_neurons)
        print("Biases:", output_layer.num_neurons)

        print("\nTotal weights:", total_weights)
        print("Total biases:", total_biases)
        print("Total parameters:", total_weights + total_biases)

        if len(self.error_history) > 0:
            print("Loss:", self.error_history[-1])

        print("----------")

        if len(self.error_history) > 0:
            plt.figure()
            plt.plot(self.error_history, label="Loss")
            plt.legend()
            plt.show()

    def evaluate(self, x, y):
        if (
            self.loss_name == "mean_squared_error"
            and self.layers[-1].activation == "linear"
        ):
            prediction = self.predict(x)
            print("Average combined error:", round(np.mean(abs(prediction - y)), 3))
            for i in range(prediction.shape[1]):
                print(
                    f"Average error for class {i + 1}:",
                    round(np.mean(abs(prediction[:, i] - y[:, i])), 3),
                )
        elif (
            self.loss_name == "binary_cross_entropy"
            and self.layers[-1].activation == "sigmoid"
        ):
            prediction = self.predict(x) >= 0.5
            print(
                "Combined accuracy:",
                str(
                    round(
                        100 * np.mean(prediction == y),
                        3,
                    )
                )
                + "%",
            )
            for i in range(prediction.shape[1]):
                print(
                    f"Accuracy for class {i + 1}:",
                    str(
                        round(
                            100 * np.mean(prediction[:, i] == y[:, i]),
                            3,
                        )
                    )
                    + "%",
                )
        elif (
            self.loss_name == "categorical_cross_entropy"
            and self.layers[-1].activation == "softmax"
        ):
            print(
                "Accuracy:",
                str(
                    round(
                        100
                        * np.mean(
                            np.argmax(self.predict(x), axis=1) == np.argmax(y, axis=1)
                        ),
                        3,
                    )
                )
                + "%",
            )
        elif (
            self.loss_name == "sparse_categorical_cross_entropy"
            and self.layers[-1].activation == "softmax"
        ):
            print(
                "Accuracy:",
                str(
                    round(
                        100 * np.mean(np.argmax(self.predict(x), axis=1) == y),
                        3,
                    )
                )
                + "%",
            )
        else:
            raise ValueError("Undefined evaluation for loss-activation combination")

    def predict(self, x):
        for weights, biases, activation in zip(
            self.weights, self.biases, self.activations
        ):
            x = activation(x @ weights + biases)
        return x

    def fit(self, x, y, epochs, learning_rate, batch_size):
        self.loss_derivative, self.derivatives[-1] = Losses.simplify(
            self.loss_name, self.layers[-1].activation
        )

        error_history = []
        num_samples = y.shape[0]

        for _ in range(epochs):
            random_indices = np.random.permutation(num_samples)[:batch_size]
            batch_x = x[random_indices]
            batch_y = y[random_indices]

            output_a = batch_x
            z_results = []
            a_results = []

            for weights, biases, activation in zip(
                self.weights, self.biases, self.activations
            ):
                a_results.append(output_a)
                z = output_a @ weights + biases
                a = activation(z)
                z_results.append(z)
                output_a = a

            error = self.loss_derivative(batch_y, output_a)

            derivatives = [
                derivative(z_result)
                for derivative, z_result in zip(self.derivatives, z_results)
            ][::-1]

            change = error * derivatives[0]
            changes = [change]

            for derivative, weights in zip(derivatives[1:], self.weights[::-1]):
                change = (change @ weights.T) * derivative
                changes.append(change)

            changes = changes[::-1]

            for index in range(len(self.weights)):
                self.biases[index] -= learning_rate * np.mean(changes[index], axis=0)
                self.weights[index] -= (
                    learning_rate * (changes[index].T @ a_results[index])
                ).T

            error_history.append(np.mean(self.loss(batch_y, output_a)))

        self.error_history = error_history

    def __add__(self, other):
        if isinstance(other, Input):
            model = Model(other.num_inputs, self.loss_name)
            for layer in self.layers:
                model.add(layer)
            return model
        if isinstance(other, Loss):
            model = Model(self.num_inputs, other.loss_name)
            for layer in self.layers:
                model.add(layer)
            return model
        if isinstance(other, Layer):
            model = Model(self.num_inputs, self.loss_name)
            for layer in self.layers:
                model.add(layer)
            model.add(other)
            return model
        if isinstance(other, Model):
            model = Model(self.num_inputs, self.loss_name)
            for layer in self.layers:
                model.add(layer)
            for layer in other.layers:
                model.add(layer)
            return model
        else:
            raise TypeError("Unsupported type for addition: {}".format(type(other)))
