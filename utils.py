import numpy as np


class Utils:
    @staticmethod
    def split_data(x, y, test_percentage=20):
        indices = np.arange(y.shape[0])
        np.random.shuffle(indices)
        split = int(y.shape[0] * test_percentage / 100)

        x = x[indices]
        y = y[indices]

        x_train = x[split:]
        y_train = y[split:]

        x_test = x[:split]
        y_test = y[:split]

        return (x_train, y_train, x_test, y_test)
