import os
import tarfile
import numpy as np
import torch
import helper.cl_dataset as cl_dataset
import helper.file_helper as file_manager
import matplotlib.pyplot as plt


__author__ = 'garrett_local'


def _binarize(labels):
    return ((labels == 0) | (labels == 1) | (labels == 8) |  # (labels == 4) |
            (labels == 9)) * 1  # * 2 - 1 (labels == 4) |

def _prepare_aiartf_data():
    train_x = torch.load('/home/wangyabin/workspace/CVPRIL/train_x.pt').numpy()
    train_y = torch.load('/home/wangyabin/workspace/CVPRIL/train_y.pt').numpy()
    test_x = torch.load('/home/wangyabin/workspace/CVPRIL/test_x.pt').numpy()
    test_y = torch.load('/home/wangyabin/workspace/CVPRIL/test_y.pt').numpy()
    # import pdb;
    # pdb.set_trace()

    # train_x = np.concatenate((train_x, test_x[:100]), axis=0)
    # train_y = np.concatenate((train_y, (test_y[:100] + 2)), axis=0)
    # train_x = np.concatenate((train_x, test_x[:100]), axis=0)
    # train_y = np.concatenate((train_y, (test_y[:100] + 4)), axis=0)
    # train_x = np.concatenate((train_x, test_x[:100]), axis=0)
    # train_y = np.concatenate((train_y, (test_y[:100] + 6)), axis=0)
    # train_x = np.concatenate((train_x, test_x[:100]), axis=0)
    # train_y = np.concatenate((train_y, (test_y[:100] + 8)), axis=0)

    train_x = train_x / np.linalg.norm(train_x, axis=1, keepdims=True) - 0.02
    test_x = test_x / np.linalg.norm(test_x, axis=1, keepdims=True) - 0.02
    return train_x, train_y, test_x, test_y


class AiartDataset(cl_dataset.CLClassDataSet):

    def __init__(self, *args, **kwargs):
        self._train_x, self._train_y, self._test_x, self._test_y = _prepare_aiartf_data()
        super(AiartDataset, self).__init__(*args, **kwargs)
        self.data_size = 0
        self.test_size = self._test_y.shape[0]
    def _original_train_x(self):
        return self._train_x

    def _original_train_y(self):
        return self._train_y

    def _original_test_x(self):
        return self._test_x

    def _original_test_y(self):
        return self._test_y
