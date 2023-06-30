import argparse
from models.classification.resnet import ResNet
from models.classification.alexnet import AlexNet
from datasets.mnist import mnist
from datasets.imagenet1k import imagenet1k
from datasets.utils import plot_learning_curves
import torch
import math
import time

def timer():
  seconds = time.time()
  return time.ctime(seconds)

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
    print("Run ResNet with " + args.data + " for " + args.pattern)
    device = torch.device('cuda')

    train_data = imagenet1k()
    val_data = imagenet1k("val")
    val_data_number = len(val_data.image_list)
    train_data_number = len(train_data.image_list)
    epoch = 100
    iter_number = epoch * train_data_number
    print("all iter number: ", iter_number)
    model = ResNet(1000).float()
    model = torch.nn.DataParallel(model)
    model.to(device)
    batch_size = 32

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.MSELoss()
    all_loss = []
    all_acc = []
    # log = open("resnet.log", "w")
    for i in range(iter_number):
      # zero the parameter gradients
      optimizer.zero_grad()
      x, y = train_data.data(batch_size)
      outputs = model.forward(torch.tensor(x, dtype=torch.float).cuda())
      labels = torch.tensor(y, dtype=torch.float).cuda()
      #print(outputs, labels)
      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()
      if (math.isnan(loss.item())): break
      # if (i % (train_data_number / batch_size) == 0):
      if (i % 1000 == 0):
        print(timer(), "iter: ", i , " loss: ", loss.item())
        all_loss.append(loss.item())
        # compute acc
        acc = 0
        miss = 0
        # for l in range(int(val_data_number / batch_size)):
        for l in range(100):
          tx, ty = val_data.data(batch_size)
          toutputs = model.forward(torch.tensor(tx, dtype=torch.float).cuda())
          label_net = torch.argmax(toutputs.cpu(), dim=1)
          label_true = torch.argmax(torch.tensor(ty, dtype=torch.float), dim=1)
          # print("test label", label_net, "\ntrue label: ", label_true)
          for k in range(len(label_net)):
            if (label_net[k] == label_true[k]):
              acc += 1
            else:
              miss += 1
        acc = acc / (acc + miss)
        all_acc.append(acc)
        torch.save(model.state_dict(), "resnet_epoch_"+str(int(i/train_data_number))+"_acc_"+str(acc)+".pt")
    plot_learning_curves("/workspace/ubuntu/model-zoo/resnet_loss.png", len(all_loss), [all_loss], ["loss"], "resnet")
    plot_learning_curves("/workspace/ubuntu/model-zoo/resnet_acc.png", len(all_acc), [all_acc], ["acc"], "resnet")
