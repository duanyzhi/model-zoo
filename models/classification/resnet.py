import torch
from torch import Tensor
import torch.nn as nn

class ResidualBlock(nn.Module):
  def __init__(self,
     input_channel: int,
     output_channel: int,
     stride: int = 1
     ):
      super(ResidualBlock, self).__init__()
      self.conv1 = nn.Conv2d(input_channel, output_channel, 3, stride = stride)
      self.bn1 = torch.nn.BatchNorm2d(output_channel)
      self.relu = nn.ReLU(inplace=True)
      self.conv2 = nn.Conv2d(output_channel, output_channel, 3)
      self.bn2 = torch.nn.BatchNorm2d(output_channel)
      self.downsample = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, 1),
                nn.BatchNorm2d(output_channel))

  def forward(self, x):
      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu(out)
      out = self.conv2(out)
      out = self.bn2(out)
      identity = self.downsample(x)
      print(out.size(), x.size(), identity.size())
      out += identity
      out = self.relu(out)
      print(out.size())
      return out
        
class ResNet(nn.Module):
  def __init__(self) -> None:
      super(ResNet, self).__init__()
      # conv1: input: [N, 3, 224, 224]
      # weight kernel: 7 * 7 * 3 * 64
      # stride = 2
      # conv1 output: [N, 64, 112, 112]
      self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2,
        padding = 0, bias = True)
      self.bn1 = torch.nn.BatchNorm2d(64)
      self.relu = nn.ReLU(inplace=True)
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

      # conv2
      self.conv2 = nn.Sequential(
        ResidualBlock(64, 64),
        ResidualBlock(64, 64),
        ResidualBlock(64, 64))
       
  def forward(self, x: Tensor) ->Tensor:
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)
      print("size after conv1: ", x.size())
      x = self.conv2(x)

      return x

def resnet(x):
  network = ResNet()
  return network.forward(x)

