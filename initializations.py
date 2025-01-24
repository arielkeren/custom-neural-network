import numpy as np


class Initializations:
    @staticmethod
    def get(name):
        match name.lower():
            case "xavier":
                return Initializations.xavier
            case "he":
                return Initializations.he
            case "zero":
                return Initializations.zero
            case _:
                return Initializations.he

    @staticmethod
    def xavier(n_in, n_out):
        return np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)

    @staticmethod
    def he(n_in, n_out):
        return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

    @staticmethod
    def zero(n_in, n_out):
        return np.zeros((n_in, n_out))
