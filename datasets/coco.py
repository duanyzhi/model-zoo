import numpy
import ijson
import os
import cv2
import random
import numpy as np

class CocoJsonParser:
  def __init__(self, root_path, mode):
    root_path = root_path + "/annotations_trainval2017/annotations/"
    self.json_path = root_path +  "/instances_train2017.json"
    if mode == "val":
      self.json_path = root_path +  "/instances_val2017.json"
    self.categories = self.parse_categories()
    self.all_ann_dict = self.annotations()
    self.annotation_dict = self.collect_info()

  def parse_categories(self):
    f = open(self.json_path)
    cat = {}
    for item in ijson.items(f, "categories.item"):
      # print("id:", item["id"], " name:", item["name"])
      cat[item["id"]] = item["name"]
    f.close()
    return cat

  def get_categories_name(self, cat_id):
    return self.categories[cat_id]

  def get_image_id(self, name):
    f = open(self.json_path)
    for item in ijson.items(f, "images.item"):
      if name == item["file_name"]:
        return item["id"]

  def annotations(self):
    f = open(self.json_path)
    all_ann_dict = {}
    i = 0
    for item in ijson.items(f, "annotations.item"):
      ann_dict = {}
      ann_dict["bbox"] = [int(x) for x in item["bbox"]]
      ann_dict["bbox"][2] = ann_dict["bbox"][0] + ann_dict["bbox"][2] 
      ann_dict["bbox"][3] = ann_dict["bbox"][1] + ann_dict["bbox"][3] 
      ann_dict["category_id"] = int(item["category_id"])
      ann_dict["category_name"] = self.get_categories_name(ann_dict["category_id"])
      all_ann_dict[item["image_id"]] = ann_dict
      i += 1
      if i == 10: break
    f.close()
    return all_ann_dict
    #  if (image_id == item["image_id"]):
    #    ann_dict["bbox"] = [int(x) for x in item["bbox"]]
    #    ann_dict["category_id"] = int(item["category_id"])
    #    ann_dict["category_name"] = self.get_categories_name(ann_dict["category_id"])
    #    break
    #f.close()
    #return ann_dict

  def collect_info(self):
    f = open(self.json_path)
    annotation_dicts = {}
    i = 0
    for item in ijson.items(f, "images.item"):
      name = item["file_name"]
      name_id = item["id"]
      if name_id in self.all_ann_dict:
        info = self.all_ann_dict[name_id]
        annotation_dicts[name] = info
        i += 1
        if i == 10:
          break
    f.close()
    return annotation_dicts
    
  def info(self, name):
    #image_id = self.get_image_id(name)
    #annotation_dict = self.annotations(image_id)
    return self.annotation_dict[name]  

  # for some image no annotation
  def find(self, name):
    if name in self.annotation_dict:
      return True
    else:
      return False

class coco:
  def __init__(self, mode):
      root_path = "/workspace/ubuntu/data/coco"
      self.parser = CocoJsonParser(root_path, "train")
      self.data_root = root_path + "/train2017/train2017"
      self.mode = mode
      if self.mode == "val":
        self.parser = CocoJsonParser(root_path, "val")
        self.data_root = root_path + "/val2017/val2017"
      self.image_list = os.listdir(self.data_root)
      self.index = 0
      self.label_len = len(self.parser.categories) + 1  # 0 for background

  def next_index(self):
      if self.index > len(self.image_list):
          self.index = 0
      else:
          self.index += 1
 
  def load(self):
      image_name = self.image_list[self.index]
      while (not self.parser.find(image_name)):
        self.next_index()
        image_name = self.image_list[self.index]
      info = self.parser.info(image_name)
      data = cv2.imread(self.data_root + "/" + image_name)
      #  st = (int(info["bbox"][0]), int(info["bbox"][1]))
      #  et = (int(info["bbox"][0] + info["bbox"][2]), int(info["bbox"][1] + info["bbox"][3]))
      #  cv2.rectangle(data, st, et, (0, 0, 255), 1)
      #  cv2.putText(data, info["category_name"], st, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
      #  cv2.imwrite("/workspace/ubuntu/model-zoo/"+self.mode+str(self.index)+"coco.png", data)
      data = np.array((data - 127.0) / 255.0)
      data = np.transpose(data, (2, 0, 1))
      print(data.shape, data.shape[0])
      data = np.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))
      info["data"] = data
      label = np.zeros((1, self.label_len))
      label[0, info["category_id"]] = 1
      info["label"] = label
      #print(data.shape)
      #print("image name:", image_name, info)
      self.next_index()
      return info
