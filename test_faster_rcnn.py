import torch
import torchvision
import sys

sys.path.insert(1, '../')
from datasets.coco import coco

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
val_data = coco("val")
input_info = val_data.load()
images = torch.tensor(input_info["data"], dtype=torch.float)
print("input_info: ", input_info)
targets = [{"boxes": torch.tensor(input_info["bbox"], dtype=torch.float).reshape(1, 4),
            "labels": torch.tensor(input_info["label"], dtype=torch.int64)}]

# For training
# images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
# labels = torch.randint(1, 91, (4, 11))
images = list(image for image in images)
# targets = []
# for i in range(len(images)):
#     d = {}
#     d['boxes'] = boxes[i]
#     d['labels'] = labels[i]
#     targets.append(d)
output = model(images, targets)

# # For inference
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)
# 
# # optionally, if you want to export the model to ONNX:
# torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)
