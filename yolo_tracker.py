from models import *
from utils.utils import *
from utils.datasets import *

import torch
import torchvision.transforms as transforms

import cv2
import sys
import argparse

from PIL import Image

import matplotlib.pyplot as plt


def TrackerInit(tracker_type):

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    return tracker

def YOLO(frame,tracker_type,img_size):
    trackers=[]
    bboxes=[]

    input_img=Image.fromarray(frame)

    trans=transforms.Compose([transforms.ToTensor()])

    input_img=trans(input_img).to(device)

    input_img, _ = pad_to_square(input_img, 0)
    input_img = resize(input_img, img_size).unsqueeze(0)

    # Detect Object
    detections = model(input_img)
    detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0]
    
    if detections is not None:
        detections=rescale_boxes(detections,opt.img_size,frame.shape[:2]).long()

        for det in detections:
            if det[-1] != 0:
                continue

            p1=tuple(det[:2])
            p2=tuple(det[2:4])
            tracker=TrackerInit(tracker_type)

            bbox=p1+p2
            ok=tracker.init(frame,bbox)

            trackers.append(tracker)
            bboxes.append(bbox)

    return trackers,bboxes

if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--model_def',type=str,default='config/yolov3.cfg')
    parser.add_argument('--weights_path',type=str,default='weights/yolov3.weights')
    parser.add_argument('--class_path',type=str,default='data/coco.names')
    parser.add_argument("--conf_thres", type=float, default=0.8)
    parser.add_argument("--nms_thres", type=float, default=0.4)
    parser.add_argument("--img_size", type=int, default=416)
    opt=parser.parse_args()

    device='cuda' if torch.cuda.is_available() else 'cpu'

    tracker_type='MEDIANFLOW'

    # YOLO model initialize
    print('YOLO model initializing...')
    model=Darknet(opt.model_def,img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()

    classes=load_classes(opt.class_path)

    video=cv2.VideoCapture('Chaplin.mp4')

    if not video.isOpened():
        print("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    count=0

    print(frame.shape)
    

    while True:
        if count%30==0:
            trackers,bboxes = YOLO(frame,tracker_type,opt.img_size)

        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        for i,t in enumerate(trackers):
            ok, bboxes[i] = t.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            for b in bboxes:
                p1 = (int(b[0]), int(b[1]))
                p2 = (int(b[2]), int(b[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

        count+=1