from models.classification.resnet import ResNet
import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Optional
from torch.nn import functional as F
import torchvision
# torch.set_printoptions(profile="full")

class Anchor:
    def __init__(self):
       super(Anchor, self).__init__()
       self.base_anchors = []
       self.generator()

    def generator(self):
       scales=[32, 128, 512]
       ratios=[0.5, 1.0, 2.0]
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
       # print("image pixel center map: ", shifts)
       # print("base anchors: ", self.base_anchors.size(), self.base_anchors)
       anchors = []
       # for base_anchor in self.base_anchors:
       #   print("base anchor", base_anchor)
       #   anchors.append((shifts.view(-1, 1, 4) + base_anchor.view(1, -1, 4)).reshape(-1, 4))
       for xy in shifts:
         anchors.append((self.base_anchors.view(1, -1, 4) + xy))
       output_anchor = torch.cat(anchors)
       # print("generator anchor: ", output_anchor.size(), output_anchor)
       # generator anchor: [N * W, number_anchor, 4]
       print("before reshape: ", output_anchor)
       output_anchor = output_anchor.reshape(-1, 4)
       print("after reshape: ", output_anchor)
       return output_anchor

def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

@torch.jit._script_if_tracing
def encode_boxes(reference_boxes, proposals, weights):
    print("anchors: ", reference_boxes)
    print("proposals: ", proposals)
    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    print("gt delta: ", targets.size(), targets)
    return targets

class RpnHead:
    def __init__(self,
       input_channel: int,
       num_anchors: int = 9,
       training: bool = False):
       super(RpnHead, self).__init__()
       self.conv1 = nn.Conv2d(input_channel, input_channel, kernel_size = 3, stride = 1, padding = 1)
       self.relu = nn.ReLU(inplace=True)
       self.cls_logits = nn.Conv2d(input_channel, num_anchors, kernel_size = 1, stride = 1)
       self.bbox_logits = nn.Conv2d(input_channel, num_anchors * 4, kernel_size = 1, stride = 1)
       self.anchor = Anchor()
       self.training = training
       self.rpn_fg_iou_thresh=0.7
       self.rpn_bg_iou_thresh=0.3
       self.weights = (1.0, 1.0, 1.0, 1.0) 

    def refine_bbox(self, offset_bbox, bbox):
       rbbox = offset_bbox.reshape(-1, 4)
       print("rbbox: ", rbbox.size())
       widths = bbox[:, 2] - bbox[:,  0]
       heights = bbox[:, 3] - bbox[:, 1]
       cx = bbox[:, 0] + 0.5 * widths
       cy = bbox[:, 1] + 0.5 * heights

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

    def clip_boxes_to_image(self, boxes: Tensor, size: List[int]) -> Tensor:
        #  format bbox to 0 to image size
        dim = boxes.dim()
        boxes_x = boxes[..., 0::2]
        boxes_y = boxes[..., 1::2]
        height, width = size[0], size[1]
    
        if torch._C._get_tracing_state():
            boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
            boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
            boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
            boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
        else:
            boxes_x = boxes_x.clamp(min=0, max=width)
            boxes_y = boxes_y.clamp(min=0, max=height)
    
        clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
        return clipped_boxes.reshape(boxes.shape)

    def remove_small_boxes(self, boxes: Tensor, min_size: float) -> Tensor:
        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        return keep

    def filter_proposals(self, proposals, cls, image_shapes : List[int]):
       # print("proposal: ", proposals.size(), cls.size())
       N, C, H, W = cls.shape
       scores = cls.permute(0, 2, 3, 1)
       scores = scores.reshape(-1, C).reshape(-1)
       bbox = proposals.reshape(-1, 4)
       # print(scores.size(), scores, bbox.size(), bbox)
       score_value, topk_index = torch.topk(scores, 200, 0)
       keep_bbox = bbox[topk_index]
       obj_prob = torch.sigmoid(score_value)
       clip_bbox = self.clip_boxes_to_image(keep_bbox, image_shapes)
       big_keep = self.remove_small_boxes(clip_bbox, 1e-3)
       score = obj_prob[big_keep]
       bbox = clip_bbox[big_keep]
       # print("keep bbox:", score, bbox)
       return score, bbox

    def box_iou(self, boxes1, gt_box):
       # boxes: [N * M, 4], N is all anchor number, M is base anchor number
       area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
       area2 = (gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])  # first dim is batch size

       lt = torch.max(boxes1[:, :2], gt_box[:, :2])
       rb = torch.min(boxes1[:, 2:], gt_box[:, 2:])
       wh = _upcast(rb - lt).clamp(min=0)

       inter = wh[:, 0] * wh[:, 1]
       union = area1 + area2 - inter
       iou = inter / union
       print("iou: ", iou)
       return iou

    def generate_bbox_labels(self, bbox, targets):
       labels = []
       gt_bbox = []
       for target in targets:
         iou = self.box_iou(bbox, target["bbox"])
         print("iou: ", iou.size(), iou)
         iou = iou.reshape((1, iou.size()[0]))
         matched_vals, matches = iou.max(dim=0)
         print("match vals: ", matched_vals, matches)
         below_lower_threshold = matched_vals < self.rpn_bg_iou_thresh
         between_thresholds = (matched_vals >= self.rpn_bg_iou_thresh) & (
                               matched_vals < self.rpn_fg_iou_thresh)
         print(below_lower_threshold)  # bg
         print(between_thresholds)  # discard

         matches[below_lower_threshold] = -1
         matches[between_thresholds] = -2
         print("matches: ", matches)
         matched_gt = target["bbox"][matches.clamp(min=0)]
         print("match gt: ", matched_gt.size(), matched_gt)

         label = matches >= 0
         label = label.to(dtype=torch.float32)
         label[below_lower_threshold] = 0.0  # bg
         label[between_thresholds] = -1.0
         print(label)
         labels.append(label)
         gt_bbox.append(matched_gt)
       return labels, gt_bbox
         
    def loss(self, anchors, targets, pre_bbox_deltas, pre_scores):
       labels, gt_bbox = self.generate_bbox_labels(anchors, targets)
       dtype = anchors.dtype
       device = anchors.device
       print("devie", device)
       weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
       gt_bbox = torch.cat(gt_bbox, dim=0)
       labels = torch.cat(labels, dim=0)
       print("xx", gt_bbox.size(), anchors.size())
       gt_delta = encode_boxes(gt_bbox, anchors, weights)
       print("loss:\n")
       print("label ", labels.size(), pre_scores.size())
       print("label ", labels, pre_scores)
       pre_bbox_deltas = pre_bbox_deltas.reshape(-1, 4)
       print("bbox: ", gt_delta.size(), pre_bbox_deltas.size())
       pre_scores = pre_scores.flatten()

       objectness_loss = F.binary_cross_entropy_with_logits(
            labels, pre_scores)
       box_loss = F.smooth_l1_loss(gt_delta, pre_bbox_deltas)
       print("loss", objectness_loss, box_loss)
       return objectness_loss, box_loss

    def forward(self, x,
                im_size : List[int],
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
        ):
        feature_size = [x.size()[2], x.size()[3]]
        x = self.conv1(x)
        x = self.relu(x)
        cls = self.cls_logits(x)
        bbox = self.bbox_logits(x)
        format_cls, pre_bbox_deltas = self.format_dims(cls, bbox)
        anchors = self.anchor.forward(feature_size, im_size)
        print("anchors: ", anchors.size(), anchors)
        print("pre cls and box: ", format_cls.size(), pre_bbox_deltas.size())
        proposals = self.refine_bbox(pre_bbox_deltas, anchors)
        scores, bbox = self.filter_proposals(proposals, cls, im_size)
        losses = {}
        if self.training:
           obj_loss, box_loss = self.loss(anchors, targets, pre_bbox_deltas, format_cls)
           losses["obj"] = obj_loss
           losses["box"] = box_loss
        return scores, bbox, losses
 
class faster_rcnn(nn.Module):
  def __init__(self):
      super(faster_rcnn, self).__init__()
      self.backbone = ResNet(feature = True)
      self.rpn = RpnHead(512, 9, True)
      c = 512 * 7 * 7
      num_classes = 91
      representation_size = 4096
      self.fc1 = nn.Linear(c, representation_size)
      self.fc2 = nn.Linear(representation_size, representation_size)

      self.cls_score = nn.Linear(representation_size, num_classes)
      self.bbox_pred = nn.Linear(representation_size, num_classes * 4)

  def convert_to_roi_format(self, boxes) -> Tensor:
      device, dtype = boxes.device, boxes.dtype
      ids = torch.cat(
          [
              torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device)
              for i, b in enumerate([boxes])
          ],
          dim=0,
      )
      rois = torch.cat([ids, boxes], dim=1)
      return rois
  
  def forward(self, x, targets=None):
    org_image_size = [x.size()[2], x.size()[3]]
    feature_map = self.backbone(x)
    rpn_cls, rpn_bbox, rpn_loss = self.rpn.forward(feature_map, org_image_size, targets)
    # print("feature map size: ", feature_map.size(), org_image_size, feature_map)
    rpn_rois = self.convert_to_roi_format(rpn_bbox)
    # print("rois: ", rpn_rois.size(), rpn_rois)
    scale = feature_map.size()[2] / org_image_size[0]
    roi_align = torchvision.ops.roi_align(feature_map, rpn_rois, 7, scale)
    # print(roi_align.size(), roi_align)
    x = roi_align.flatten(start_dim=1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))

    cls = self.cls_score(x)
    box = self.bbox_pred(x)
    print(cls.size(), box.size())
    print(rpn_cls.size())
