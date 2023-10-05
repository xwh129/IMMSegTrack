from ultralytics import YOLO
import motmetrics as mm
import numpy as np 
import os
from utils_ds.io import read_results

# settings of hyper-parameters
List=[[0.976,0.99,0.821]]
# loading model
result_dict=dict()
max_score=0

signal=1
while signal<=49:
    if (os.path.exists("datasets/SNMOT/"+str(signal))):
        for i_l in range(len(List)):
            # 使用模型
            model = YOLO("yolov8x-seg.pt")  # loading model
            model.track("datasets/SNMOT/"+str(signal)+"/1.mp4", save_txt=True, save=True, thres_control=List[i_l],
                        stream=True)  # predict
            print("//////////////////////////////////////////////////////////////////////////////////////////////////",
                  2 + i_l)
    signal+=1
