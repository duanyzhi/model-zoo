import argparse
from models.classification.resnet import resnet
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
    print("Run ResNet with " + args.data + " for " + args.pattern)
    x = torch.rand(1, 3, 224, 224, dtype=torch.float32, device='cpu')
    out = resnet(x)
