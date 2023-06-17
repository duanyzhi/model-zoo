import struct
import random
import tarfile
import cv2
import os
import sys
import pickle as p
import numpy as np

import matplotlib.pyplot as plt

np.set_printoptions(suppress=True,linewidth=sys.maxsize,threshold=sys.maxsize)

def loadImageSet(binfile):
    buffers = binfile.read()
    head = struct.unpack_from('>IIII', buffers, 0)
    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'  # like '>47040000B'
    imgs = struct.unpack_from(bitsString, buffers, offset)
    binfile.close()
    imgs = np.reshape(imgs, [imgNum, 1, width * height])
    return imgs


def loadLabelSet(binfile):
    buffers = binfile.read()
    head = struct.unpack_from('>II', buffers, 0)
    imgNum = head[1]
    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])
    return labels

def bn(batch_data, gamma=1, beta=0):
    mean = np.mean(batch_data)
    std = np.std(batch_data)
    # Use adjusted standard deviation here, in case the std == 0.
    # std = np.max([np.std(batch_data), 1.0/np.sqrt(batch_data.shape[1]*batch_data.shape[2]*batch_data.shape[3])])
    batch_norm_data = (batch_data - mean) / std
    return gamma*batch_norm_data + beta

class mnist:
    def __init__(self):
        root_path = "/workspace/ubuntu/data/mnist"
        self.train_images_in = open(root_path + "/train-images.idx3-ubyte", 'rb')
        self.train_labels_in = open(root_path + "/train-labels.idx1-ubyte", 'rb')
        self.test_images_in = open(root_path + "/t10k-images.idx3-ubyte", 'rb')
        self.test_labels_in = open(root_path + "/t10k-labels.idx1-ubyte", 'rb')
        self.batch_size = 32
        self.train_image = loadImageSet(self.train_images_in)  # [60000, 1, 784]
        self.train_labels = loadLabelSet(self.train_labels_in)  # [60000, 1]
        self.test_images = loadImageSet(self.test_images_in)  # [10000, 1, 784]
        self.test_labels = loadLabelSet(self.test_labels_in)  # [10000, 1]
        self.data = {"train": self.train_image, "test": self.test_images}
        self.label = {"train": self.train_labels, "test": self.test_labels}
        self.indexes = {"train": 0, "val": 0, "test": 0}

    def get_mini_batch(self, data_name="train"):
        if (self.indexes[data_name] + 1) * self.batch_size > self.data[data_name].shape[0]:
            self.indexes[data_name] = 0
        batch_data = self.data[data_name][
                     self.indexes[data_name] * self.batch_size:(self.indexes[data_name] + 1) * self.batch_size, :, :]
        batch_label = self.label[data_name][
                      self.indexes[data_name] * self.batch_size:(self.indexes[data_name] + 1) * self.batch_size, :]
        test_data = np.reshape(batch_data[0], (28, 28))
        print(test_data)
        self.indexes[data_name] += 1
        y = np.zeros((self.batch_size, 10))
        for kk in range(self.batch_size):
            y[kk, int(batch_label[kk])] = 1.0
        x = bn(batch_data)
        x = np.reshape(x, (self.batch_size, 1, 784))
        x = np.reshape(x, (self.batch_size, 1, 28, 28))
        return x, y
