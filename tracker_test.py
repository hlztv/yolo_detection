import cv2
import os

path='outputs/'

tracker=cv2.TrackerKCF_create()

img=cv2.imread('outputs/0.png')

bbox=cv2.selectROI(img, False)
print(bbox)

ok = tracker.init(img, bbox)

for i in range(1,448):
    img_path=path+str(i)+'.png'

    img=cv2.imread(img_path)

    ok,bbox=tracker.update(img)
    
    p1=(int(bbox[0]),int(bbox[1]))
    p2=(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))
    cv2.rectangle(img,p1,p2,(255,0,0),2,1)

    cv2.imshow('track',img)
    cv2.waitKey(50)