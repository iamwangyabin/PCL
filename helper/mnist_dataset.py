from sklearn.datasets import fetch_openml

import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
import torchvision
import pdb
import helper.cl_dataset as cl_dataset

__author__ = 'garrett_local'


def _prepare_mnist_data():
    mnist = fetch_openml('mnist_784', data_home='./Data/')
    x = mnist.data
    y = mnist.target
    x = np.reshape(x, (x.shape[0], -1)) / 255.0 / 10.0
    x = x / np.linalg.norm(x, axis=1, keepdims=True) - 0.02
    train_x = np.asarray(x[:60000], dtype=np.float32)
    train_y = np.asarray(y[:60000], dtype=np.int32)
    test_x = np.asarray(x[60000:], dtype=np.float32)
    test_y = np.asarray(y[60000:], dtype=np.int32)

    return train_x, train_y, test_x, test_y


class MnistDataset(cl_dataset.CLClassDataSet):

    def __init__(self, *args, **kwargs):
        self._train_x, self._train_y, self._test_x, self._test_y = \
            _prepare_mnist_data()
        super(MnistDataset, self).__init__(*args, **kwargs)
        self.test_size = self._test_y.shape[0]
    def _original_train_x(self):
        return self._train_x

    def _original_train_y(self):
        return self._train_y

    def _original_test_x(self):
        return self._test_x

    def _original_test_y(self):
        return self._test_y
