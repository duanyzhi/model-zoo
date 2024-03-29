import argparse
from models.classification.resnet import ResNet
from models.classification.alexnet import AlexNet
from datasets.mnist import mnist
from datasets.imagenet1k import imagenet1k
from datasets.utils import plot_learning_curves
import torch
import math

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
    data = mnist()
    # imagenet_data = imagenet1k()
    network = AlexNet().float()
    # network = ResNet(1000).float()

    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.MSELoss()
    all_loss = []
    all_acc = []
    for i in range(0):
      # zero the parameter gradients
      optimizer.zero_grad()
      x, y = data.get_mini_batch()
      outputs = network.forward(torch.tensor(x, dtype=torch.float))
      labels = torch.tensor(y, dtype=torch.float)
      #print(outputs, labels)
      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()
      print("loss: ", loss.item())
      if (math.isnan(loss.item())): break
      if (i % 20 == 0):
        all_loss.append(loss.item())
        # compute acc
        acc = 0
        miss = 0
        for l in range(10):
          tx, ty = data.get_mini_batch("test")
          toutputs = network.forward(torch.tensor(tx, dtype=torch.float))
          label_net = torch.argmax(toutputs, dim=1)
          label_true = torch.argmax(torch.tensor(ty, dtype=torch.float), dim=1)
          print("test label", label_net, label_true)
          for k in range(len(label_net)):
            if (label_net[k] == label_true[k]):
              acc += 1
            else:
              miss += 1
        acc = acc / (acc + miss)
        all_acc.append(acc)
    plot_learning_curves("/workspace/ubuntu/model-zoo/loss.png", len(all_loss), [all_loss], ["loss"], "alexnet")
    plot_learning_curves("/workspace/ubuntu/model-zoo/acc.png", len(all_acc), [all_acc], ["acc"], "alexnet")
    torch.save(network.state_dict(), "alexnet_mnist.pt")
