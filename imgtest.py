import cv2
import numpy as np
img=cv2.imread('img.jpg')
net=cv2.dnn.readNetFromTensorflow('frozen_inference_graph_coco.pb','mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')
img=cv2.resize(img,(1020,600))
overlay=img.copy()
h,w,_=img.shape
class_names=[]
with open("coco.txt",'r') as f:
    class_names=f.read().split()
blob=cv2.dnn.blobFromImage(img,swapRB=True)
net.setInput(blob)
boxes,masks=net.forward(['detection_out_final','detection_masks'])
count_of_boxes=boxes.shape[2]
#print(count_of_boxes)
for i in range(count_of_boxes):
    box=boxes[0,0,i]
#print(box)
    x=int(box[3]*w)
    y=int(box[4]*h)
    x1=int(box[5]*w)  
    y1=int(box[6]*h)
    class_id=box[1]
    class_id=int(class_id)
    score=box[2]
    labels=class_names[class_id]
    if score > 0.0:
#    print(x,y,x1,y1)
       cv2.rectangle(img,(x,y),(x1,y1),(0,0,255),3)
       cv2.putText(img,str(labels),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2)
       roi=img[y:y1,x:x1]
       roi_h,roi_w,_=roi.shape
       mask=masks[1,class_id]
       mask=cv2.resize(mask,(roi_w,roi_h))
       _,mask=cv2.threshold(mask,0.7,255,cv2.THRESH_BINARY)
       contours,_=cv2.findContours(np.array(mask,np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
       cv2.drawContours(roi, contours, -1, (0,255,0), 2)
       cv2.fillPoly(roi,contours,(255,0,255))
#       cv2.imshow(str(x*y),roi)
       
       img_new=cv2.addWeighted(overlay,0.5,img,1 - 0.5,0)
       cv2.imshow('MASK',img_new)
#cv2.imshow("IMG",img)
cv2.waitKey(0)
