#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader


class YOLO_Pred():
    def __init__(self,onnx_model,data_yaml):
        # load YAML
        with open(data_yaml,mode='r') as f:
            data_yaml = yaml.load(f,Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        
        # load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Pre-compute colors for better performance
        np.random.seed(10)
        self.colors = np.random.randint(100,255,size=(self.nc,3)).tolist()
    
    def predictions(self,image):
        row, col, d = image.shape
        # get the YOLO prediction from the the image
        # step-1 convert image into square image (array)
        max_rc = max(row,col)
        input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
        input_image[0:row,0:col] = image
        
        # step-2: get prediction from square array
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WH_YOLO,INPUT_WH_YOLO),swapRB=True,crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward() # detection or prediction from YOLO

        # Non Maximum Supression
        # step-1: filter detection based on confidence (0.4) and probability score (0.25)
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # widht and height of the image (input_image)
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w/INPUT_WH_YOLO
        y_factor = image_h/INPUT_WH_YOLO

        # Vectorized operations for better performance
        confidence_mask = detections[:, 4] > 0.4
        if confidence_mask.any():
            class_scores = detections[confidence_mask, 5:]
            class_ids = np.argmax(class_scores, axis=1)
            max_scores = np.max(class_scores, axis=1)
            score_mask = max_scores > 0.25
            
            if score_mask.any():
                valid_detections = detections[confidence_mask][score_mask]
                valid_class_ids = class_ids[score_mask]
                
                # Convert to numpy arrays for faster processing
                boxes = []
                confidences = []
                classes = []
                
                for i, (det, class_id) in enumerate(zip(valid_detections, valid_class_ids)):
                    cx, cy, w, h = det[0:4]
                    left = int((cx - 0.5*w)*x_factor)
                    top = int((cy - 0.5*h)*y_factor)
                    width = int(w*x_factor)
                    height = int(h*y_factor)
                    
                    boxes.append([left, top, width, height])
                    confidences.append(det[4])
                    classes.append(class_id)

        # Convert to numpy arrays for NMS
        boxes_np = np.array(boxes)
        confidences_np = np.array(confidences)

        # NMS
        if len(boxes_np) > 0:
            index = cv2.dnn.NMSBoxes(boxes_np.tolist(), confidences_np.tolist(), 0.25, 0.45).flatten()

            # Draw the Bounding
            for ind in index:
                # extract bounding box
                x,y,w,h = boxes_np[ind]
                bb_conf = int(confidences_np[ind]*100)
                classes_id = classes[ind]
                class_name = self.labels[classes_id]
                color = tuple(self.colors[classes_id])

                text = f'{class_name}: {bb_conf}%'

                cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
                cv2.rectangle(image,(x,y-30),(x+w,y),color,-1)

                cv2.putText(image,text,(x,y-10),cv2.FONT_HERSHEY_PLAIN,0.7,(0,0,0),1)
            
        return image
    
    
    def generate_colors(self,ID):
        np.random.seed(10)
        colors = np.random.randint(100,255,size=(self.nc,3)).tolist()
        return tuple(colors[ID])
        
        
    
    
    



