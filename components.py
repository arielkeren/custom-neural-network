from layer import Layer


class Input:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs

    def __add__(self, other):
        from model import Model

        if isinstance(other, Input):
            raise TypeError("Input layer cannot be added to another input layer")
        if isinstance(other, Loss):
            return Model(self.num_inputs, other.loss_name)
        if isinstance(other, Layer) or isinstance(other, Model):
            return other + self
        else:
            raise TypeError("Unsupported type for addition: {}".format(type(other)))


class Loss:
    def __init__(self, loss_name):
        self.loss_name = loss_name

    def __add__(self, other):
        from model import Model

        if isinstance(other, Input):
            return Model(other.num_inputs, self.loss_name)
        if isinstance(other, Loss):
            raise TypeError("Loss layer cannot be added to another loss layer")
        if isinstance(other, Layer) or isinstance(other, Model):
            return other + self
        else:
            raise TypeError("Unsupported type for addition: {}".format(type(other)))


class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__("mean_squared_error")


class BinaryCrossEntropy(Loss):
    def __init__(self):
        super().__init__("binary_cross_entropy")


class CategoricalCrossEntropy(Loss):
    def __init__(self):
        super().__init__("categorical_cross_entropy")


class SparseCategoricalCrossEntropy(Loss):
    def __init__(self):
        super().__init__("sparse_categorical_cross_entropy")


class Relu(Layer):
    def __init__(self, num_neurons, initialization="he"):
        super().__init__(num_neurons, "relu", initialization)


class Sigmoid(Layer):
    def __init__(self, num_neurons, initialization="xavier"):
        super().__init__(num_neurons, "sigmoid", initialization)


class Tanh(Layer):
    def __init__(self, num_neurons, initialization="xavier"):
        super().__init__(num_neurons, "tanh", initialization)


class Softmax(Layer):
    def __init__(self, num_neurons, initialization="xavier"):
        super().__init__(num_neurons, "softmax", initialization)


class Linear(Layer):
    def __init__(self, num_neurons, initialization="xavier"):
        super().__init__(num_neurons, "linear", initialization)
