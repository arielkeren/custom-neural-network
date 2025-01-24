class Layer:
    def __init__(self, num_neurons, activation, initialization):
        self.num_neurons = num_neurons
        self.activation = activation
        self.initialization = initialization

    def __add__(self, other):
        from model import Model
        from components import Input, Loss

        if isinstance(other, Input):
            model = Model(other.num_inputs, "binary_cross_entropy")
            model.add(self)
            return model
        if isinstance(other, Loss):
            model = Model(2, other.loss_name)
            model.add(self)
            return model
        if isinstance(other, Layer):
            model = Model(2, "binary_cross_entropy")
            model.add(self)
            model.add(other)
            return model
        if isinstance(other, Model):
            model = Model(other.num_inputs, other.loss_name)
            model.add(self)
            for layer in other.layers:
                model.add(layer)
            return model
        else:
            raise TypeError("Unsupported type for addition: {}".format(type(other)))
