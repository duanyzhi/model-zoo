import argparse
from datasets.coco import coco
from models.detection.faster_rcnn import faster_rcnn
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern',
                        type=str,
                        default='test',
                        required=False,
                        help='Choice train or test model')
    parser.add_argument('--data',
                        type=str,
                        default='cifar10',
                        required=False,
                        help='Choice which dataset')
    args = parser.parse_args()
    val_data = coco("val")
    # train_data = coco("train")
    network = faster_rcnn().float()

    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.MSELoss()
    for iter in range(1):
      input_info = val_data.load()
      input_data = torch.tensor(input_info["data"], dtype=torch.float)
      print("input_info: ", input_info)
      targets = [{"bbox": torch.tensor(input_info["bbox"], dtype=torch.float).reshape(1, 4),
                  "label": torch.tensor(input_info["label"])}]
      network.forward(input_data, targets)
      #train_data.load()
 
