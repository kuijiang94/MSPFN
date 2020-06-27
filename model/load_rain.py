import numpy as np

def load():
    x_train = np.load('train_data/npy/train_gt.npy')
    x_test = np.load('train_data/npy/test_gt.npy')
    x_train_rain = np.load('train_data/npy/train_rain.npy')
    x_test_rain = np.load('train_data/npy/test_rain.npy')
    return x_train, x_test, x_train_rain, x_test_rain

