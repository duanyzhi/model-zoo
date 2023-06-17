from models.classification.resnet import ResNet
import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Optional
torch.set_printoptions(profile="full")

class Anchor:
    def __init__(self):
       super(Anchor, self).__init__()
       self.base_anchors = []
       self.generator()

    def generator(self):
       scales=[128, 256]
       ratios=[1]
       # for w in sizes:
       #   for r in ratios:
       #     h = w * r
       #     self.base_anchors.append(torch.as_tensor(-w * 0.5, -h * 0.5, w * 0.5, h * 0.5))
       scales = torch.as_tensor(scales, dtype=torch.float32)
       aspect_ratios = torch.as_tensor(ratios, dtype=torch.float32)
       h_ratios = torch.sqrt(aspect_ratios)
       w_ratios = 1 / h_ratios
       ws = (w_ratios[:, None] * scales[None, :]).view(-1)
       hs = (h_ratios[:, None] * scales[None, :]).view(-1)
       self.base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
       self.base_anchors = self.base_anchors.round()
           
    def forward(self, image_size: List[int], org_image_size: List[int]):
       sw = org_image_size[0] / image_size[0]  # scale from org image to feature
       sh = org_image_size[1] / image_size[1]
       grid_width = image_size[0]
       grid_height = image_size[1]
       shifts_x = torch.arange(0, grid_width, dtype=torch.int32) * sw
       shifts_y = torch.arange(0, grid_height, dtype=torch.int32) * sh
       shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
       shift_x = shift_x.reshape(-1)
       shift_y = shift_y.reshape(-1)
       shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
       print("image pixel center map: ", shifts)
       print("base anchors: ", self.base_anchors.size(), self.base_anchors)
       anchors = []
       # for base_anchor in self.base_anchors:
       #   print("base anchor", base_anchor)
       #   anchors.append((shifts.view(-1, 1, 4) + base_anchor.view(1, -1, 4)).reshape(-1, 4))
       for xy in shifts:
         print("xy", xy)
         anchors.append((self.base_anchors.view(1, -1, 4) + xy))
       output_anchor = torch.cat(anchors)
       print("generator anchor: ", output_anchor.size(), output_anchor)
       # generator anchor: [N * W, number_anchor, 4]
       return output_anchor

       
class RpnHead:
    def __init__(self,
       input_channel: int,
       num_anchors: int = 2):
       super(RpnHead, self).__init__()
       self.conv1 = nn.Conv2d(input_channel, input_channel, kernel_size = 3, stride = 1, padding = 1)
       self.relu = nn.ReLU(inplace=True)
       self.cls_logits = nn.Conv2d(input_channel, num_anchors, kernel_size = 1, stride = 1)
       self.bbox_logits = nn.Conv2d(input_channel, num_anchors * 4, kernel_size = 1, stride = 1)
       self.anchor = Anchor()

    def refine_bbox(self, offset_bbox, bbox):
       rbbox = offset_bbox.reshape(-1, offset_bbox.shape[2])
       widths = bbox[:, :, 2] - bbox[:, :,  0]
       heights = bbox[:, :, 3] - bbox[:, :, 1]
       cx = bbox[:, :, 0] + 0.5 * widths
       cy = bbox[:, :, 1] + 0.5 * heights

       dx = rbbox[:, 0::4]
       dy = rbbox[:, 1::4]
       dw = rbbox[:, 2::4]
       dh = rbbox[:, 3::4]

       pre_cx = dx * widths + cx
       pre_cy = dy * heights + cy
       pre_w = torch.exp(dw) * widths
       pre_h = torch.exp(dh) * heights

       x0 = pre_cx - pre_w * 0.5
       y0 = pre_cy - pre_h * 0.5
       x1 = pre_cx + pre_w * 0.5
       y1 = pre_cy + pre_h * 0.5
       pre_bbox = torch.stack((x0, y0, x1, y1), dim=2)
       return pre_bbox
       
    def permute_and_flatten(self, layer, N, C, H, W):
        # type: (Tensor, int, int, int, int, int) -> Tensor
        layer = layer.view(N, -1, C, H, W)
        layer = layer.permute(0, 3, 4, 1, 2)
        layer = layer.reshape(N, -1, C)
        # if number Anchor = A
        # cls output size: [N, H * W, A]
        # bbox output size: [N, H * W, 4 * A]
        # [0, 0, :] -> pixel (0, 0) all anchor info 
        # [0, 1, :] -> pixel (0, 1)
        return layer

    def format_dims(self, cls, bbox):
       #print("input size: ", cls.size(), bbox.size())
       N, C, H, W = cls.shape
       # print("N, C, H, W", N, C, H, W)
       #print("org cls: ", cls)
       cls_format = self.permute_and_flatten(cls, N, C, H, W)
       bbox_format = self.permute_and_flatten(bbox, N, 4 * C, H, W)
       # print(cls_format.size())
       # print("format cls: ", cls_format)
       # print("org bbox: ", bbox)
       # print("format bbox: ", bbox_format.size(), bbox_format)
       return cls_format, bbox_format

    def filter_proposals(self, proposals, cls, image_shapes):
       print("proposal: ", proposals.size(), cls.size())
       N, C, H, W = cls.shape
       scores = cls.permute(0, 2, 3, 1)
       scores = scores.reshape(-1, C).reshape(-1)
       bbox = proposals.reshape(-1, 4)
       print(scores.size(), scores, bbox.size(), bbox)
       score_value, topk_index = torch.topk(scores, 1000, 0)
       print(score_value, topk_index)
       keep_bbox = bbox[topk_index]
       print("keep bbox:", keep_bbox)
       return score_value, keep_bbox

    def forward(self, x, im_size : List[int]):
        feature_size = [x.size()[2], x.size()[3]]
        x = self.conv1(x)
        x = self.relu(x)
        cls = self.cls_logits(x)
        bbox = self.bbox_logits(x)
        format_cls, format_bbox = self.format_dims(cls, bbox)
        anchors = self.anchor.forward(feature_size, im_size)
        proposals = self.refine_bbox(format_bbox, anchors)
        self.filter_proposals(proposals, cls, im_size)
        return format_cls, proposals
 
class faster_rcnn(nn.Module):
  def __init__(self):
      super(faster_rcnn, self).__init__()
      self.backbone = ResNet(feature = True)
      self.rpn = RpnHead(512)

  def forward(self, x):
    org_image_size = [x.size()[2], x.size()[3]]
    feature_map = self.backbone(x)
    rpn_cls, rpn_bbox = self.rpn.forward(feature_map, org_image_size)
