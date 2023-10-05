import shutil

import cv2
import os

def rm_folder(folder: str):
    folder = os.path.abspath(folder)
    if os.path.isdir(folder):
        try:
            shutil.rmtree(folder)
            print(f"delete succeeded, folder:{folder}")
        except FileNotFoundError as e:
            print(f"target doesn't exist, folder:{folder}, e:{e}")
        except Exception as e:
            print(f"delete failed, folder:{folder}, e:{e}")

def rm_file(filename: str):
    filename = os.path.abspath(filename)
    if os.path.isfile(filename):
        try:
            os.remove(filename)
            print(f"delete succeeded, filename:{filename}")
        except FileNotFoundError:
            print(f"target doesn't exist, filename:{filename}")
        except Exception as e:
            print(f"delete failed, filename:{filename}, e:{e}")

fourcc = cv2.VideoWriter_fourcc("m", "p","4","v")



signal=1
while signal<=49:
    if(os.path.exists("databak/seg_kalman_x_segment/track"+str(signal)+"/"+"imgs")):
        rm_folder("databak/seg_kalman_x_segment/track"+str(signal)+"/"+"imgs")
    signal+=1