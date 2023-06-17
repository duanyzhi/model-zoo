import struct
import random
import tarfile
import cv2
import os
import sys
import pickle as p
import numpy as np
import ijson
import json
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True,linewidth=sys.maxsize,threshold=sys.maxsize)

class imagenet1k:
    def __init__(self, mode = "train"):
        self.root_path = "/workspace/ubuntu/data/imagenet1k"
        self.labels = self.load_labels()
        self.data_path = self.root_path + "/train"
        if mode == "val":
          self.data_path = self.root_path + "/val"
        self.image_list = os.listdir(self.data_path)
        self.index = 0
        self.label_length = len(self.labels)

    def next_index(self, batch_size):
      if self.index + 2 * batch_size > len(self.image_list):
          self.index = 0
      else:
          self.index += batch_size
 
    def load_labels(self):
       label_path = self.root_path + "/labels.json" 
       f = open(label_path)
       labels = {}
       data = json.load(f)
       for count, key in enumerate(data.keys()):
         labels[key] = (count, data[key])
       return labels

    def search_label(self, im_name):
      infos = im_name.split(".")[0].split("_")
      return self.labels[infos[len(infos) - 1]][0]

    def data(self, batch_size):
       batch_list = self.image_list[self.index : self.index + batch_size]
       batch_np_data = [];
       batch_np_labels = []
       for name in batch_list:
         label_index = self.search_label(name)
         label = np.zeros((1, self.label_length))
         label[0, label_index] = 1
         batch_np_labels.append(label)
         org = cv2.imread(self.data_path + "/" + name)
         data = cv2.resize(org, (224, 224), interpolation = cv2.INTER_AREA)
         data = np.array((data - 127.0) / 255.0)
         data = np.transpose(data, (2, 0, 1))
         data = np.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))
         batch_np_data.append(data)
       batch_labels = np.concatenate(batch_np_labels, 0)
       batch_data = np.concatenate(batch_np_data, 0)
       self.next_index(batch_size)
       return batch_data, batch_labels
 
       
        
