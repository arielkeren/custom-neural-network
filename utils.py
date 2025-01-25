import numpy as np
import matplotlib.pyplot as plt


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

    @staticmethod
    def one_hot_encode(y):
        num_classes = np.max(y) + 1
        encoded = np.zeros((y.shape[0], num_classes), dtype=int)
        encoded[np.arange(y.shape[0]), y] = 1
        return encoded

    @staticmethod
    def one_hot_decode(y):
        return np.argmax(y, axis=1)

    @staticmethod
    def plot_loss(loss, val_loss=None):
        plt.plot(loss, label="Training Loss")
        if val_loss:
            plt.plot(val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
