#!/usr/bin/python3

import argparse
import os
import platform
import sys
from pathlib import Path

imageDir= os.path.abspath('images')
expDir = os.path.abspath('exp')
source = ['1.jpg',
          '2.jpg',
          '3.jpg',
          '4.jpg',
          '5.jpg',
          '6.jpg',
          '7.jpg',
          '8.png',
          '9.jpg',
          '10.jpeg',]
# Path to yolov5
yolov5Dir = os.path.abspath(str(Path.home()) + '/yolov5')
sys.path.insert(0, yolov5Dir)
os.chdir(imageDir)
print(f"Current working directory: {os.getcwd()}")

import torch

from models.common import DetectMultiBackend

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box

from utils.torch_utils import select_device, smart_inference_mode

visualize = False
augment=False
device = ''
weights = 'yolov5/yolov5s.pt'
imgsz=(640, 640)
max_det=500
conf_thres=0.5
iou_thres=0.45
classes=None
agnostic_nms=False
save_img = True
line_thickness=2
hide_labels=False
hide_conf=False

device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=False, data='data/coco128.yaml', fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)
bs = 1

dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt,)
# vid_path, vid_writer = [None] * bs, [None] * bs
# print(dataset)

model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
for path, im, im0s, vid_cap, s in dataset:
    # print(path, im)
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    for i, det in enumerate(pred):
        seen += 1
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(expDir + '/' + p.name)  # im.jpg
        # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                im0 = annotator.result()
                if save_img:      # Add bbox to image
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
    print(save_path)
    cv2.imwrite(save_path, im0)
