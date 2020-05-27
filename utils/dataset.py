import numpy as np
def pretrain_dataset():
    return np.load('weights/train.npy')
def load_labels():
    return np.load('weights/labels.npy')
def save_dataset(data):
    return np.save('weights/train.npy', data)
def save_labels(labels):
    return np.save('weights/labels.npy',labels)