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
      self.conv1 = nn.Conv2d(input_channel, output_channel, 3, stride = stride, padding=1)
      self.bn1 = torch.nn.BatchNorm2d(output_channel)
      self.relu = nn.ReLU(inplace=True)
      self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
      self.bn2 = torch.nn.BatchNorm2d(output_channel)
      self.downsample = None
      if stride != 1 or input_channel != output_channel:
        self.downsample = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(output_channel))

  def forward(self, x):
      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu(out)
      out = self.conv2(out)
      out = self.bn2(out)
      identity = x
      if self.downsample is not None:
        identity = self.downsample(x)
      # print("debug before += ", out.size(), x.size(), identity.size())
      out += identity
      out = self.relu(out)
      return out

# resnet 34
class ResNet(nn.Module):
  def __init__(self,
      cls_number: int = 10,
      feature: bool = False
      ) -> None:
      super(ResNet, self).__init__()
      # conv1: input: [N, 3, 224, 224]
      # weight kernel: 7 * 7 * 3 * 64
      # stride = 2
      # conv1 output: [N, 64, 112, 112]
      self.only_feature = feature
      self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2,
        padding = 3, bias = True)
      self.bn1 = torch.nn.BatchNorm2d(64)
      self.relu = nn.ReLU(inplace=True)
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

      # conv2
      self.conv2 = nn.Sequential(
        ResidualBlock(64, 64),
        ResidualBlock(64, 64),
        ResidualBlock(64, 64))

      self.conv3 = nn.Sequential(
        ResidualBlock(64, 128, 2),
        ResidualBlock(128, 128),
        ResidualBlock(128, 128),
        ResidualBlock(128, 128))

      self.conv4 = nn.Sequential(
        ResidualBlock(128, 256, 2),
        ResidualBlock(256, 256),
        ResidualBlock(256, 256),
        ResidualBlock(256, 256),
        ResidualBlock(256, 256),
        ResidualBlock(256, 256))
       
      self.conv5 = nn.Sequential(
        ResidualBlock(256, 512, 2),
        ResidualBlock(512, 512),
        ResidualBlock(512, 512))

      self.avgpool = nn.AdaptiveAvgPool2d((1, 1)); 
      self.fc = nn.Linear(512, cls_number)
 
  def forward(self, x: Tensor) ->Tensor:
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)
      x = self.conv2(x)
      x = self.conv3(x)
      x = self.conv4(x)
      x = self.conv5(x)

      if self.only_feature:
        return x
      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.fc(x)
      return x
