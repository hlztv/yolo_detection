from models import *
from utils.utils import *
from utils.datasets import *

import torch
import torchvision.transforms as transforms

import cv2
import sys
import argparse

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

def YOLO(frame,img_size,is_cuda):

    input_img=Image.fromarray(frame)

    trans=transforms.Compose([transforms.ToTensor()])

    input_img=trans(input_img).unsqueeze(0)

    if is_cuda:
        input_img=input_img.to('cuda')

    # Detect Object
    detections = model(input_img)
    detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0]
    
    if detections is not None:
        for det in detections:
            if det[-1] != 0:
                continue

            p1=tuple(det[:2])
            p2=tuple(det[2:4])

            bbox=p1+p2
            return bbox

def pad_to_square_numpy(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img=cv2.copyMakeBorder(img,pad[0],pad[1],pad[2],pad[3],cv2.BORDER_CONSTANT,value=pad_value)

    return img

def resize_numpy(img,img_size):
    #img=pad_to_square_numpy(img,0)
    img=cv2.resize(img, dsize=(img_size,img_size),interpolation=cv2.INTER_NEAREST)
    
    return img

if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--model_def',type=str,default='config/yolov3-tiny.cfg')
    parser.add_argument('--weights_path',type=str,default='weights/yolov3-tiny.weights')
    parser.add_argument('--class_path',type=str,default='data/coco.names')
    parser.add_argument("--conf_thres", type=float, default=0.4)
    parser.add_argument("--nms_thres", type=float, default=0.4)
    parser.add_argument("--img_size", type=int, default=416)
    opt=parser.parse_args()

    device=True if torch.cuda.is_available() else False

    tracker_type='KCF'

    print('YOLO model initializing...')
    model=Darknet(opt.model_def,img_size=opt.img_size)

    if device:
        model=model.to('cuda')

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()

    classes=load_classes(opt.class_path)
    tracker = cv2.TrackerKCF_create()

    count=1

    while True:
        # Read a new frame
        origin_frame = cv2.imread('outputs/'+str(count)+'.png')

        # if not ok:
        #     break

        frame=resize_numpy(origin_frame,opt.img_size)

        if count%30 == 0 :
            bbox = YOLO(frame,opt.img_size,device)
            ok=tracker.init(frame,bbox)
        else:
            ok,bbox=tracker.update(frame)

        # Start timer
        timer = cv2.getTickCount()

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        
        bbox=rescale_boxes([list(bbox)],opt.img_size,origin_frame.shape[:2])[0]

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(origin_frame, p1, p2, (0,255,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(origin_frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        cv2.putText(origin_frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
     
        # Display FPS on frame
        cv2.putText(origin_frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display result
        cv2.imshow("Tracking", origin_frame)
        cv2.imshow('mini',frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

        count+=1