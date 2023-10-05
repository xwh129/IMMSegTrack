# IMMSegTrack
A well-designed tracker aiming to perform better in datasets with features like irregular motions and complex postures for multiple object tracking (MOT) in computer vision. Further experiments are needed to examine the effectiveness of our proposed method.
## Introduction
The proposed solution treat ByteTrack as baseline, with the model from YOLOv8 used for segmentation. The innovations in this solution are as follows: 
- Segmentation contours matching for object tracking: This approach improves the accuracy of matching before updating the motion models compared to the traditional IoU matching method.
- Interactive Multiple Model (IMM) filtering for motion modeling: Instead of using traditional linear Kalman filtering, this solution adopts IMM filtering to model the objects' motions, resulting in tracking boxes that better fit the target's motion trajectory.
- Improved matching strategy in ByteTrack: Low-confidence unmatched detection boxes are re-evaluated as the second one in ByteTrack by comparing them with unmatched trackers to enhance the matching process.
- Reducing buffer frames when the tracked object is at the brim of the image boundary: To optimize performance, the solution removes corresponding trajectories when the tracked object reaches the image boundary.
These innovations aim to enhance the performance and accuracy of multiple object tracking in the given scenarios.
## Quick Start
- "requirements.txt" is not provided in this project, please set up environment by following the establishing guides in ByteTrack (https://github.com/ifzhang/ByteTrack.git) and YOLOv8 (https://github.com/ultralytics/ultralytics.git), and make sure that Shapely and Sympy are contained in library. 
- After the establishment of necessary environments, you should prepared datasets (in MOT20 format) and models in a folder located at root directory and run `python seg_track_release.py` to obtain evaluation labels for designated videos. 
- Then, run `python eval_track.py` and `python eval_all.py` to obtain the results in CLEAR MOT Metrics of the processed labels. 
- In addition, you should look through `ultralytics/tracker/trackers` and change the tracker in `ultralytics/tracker/trackers/__init__.py` before evaluation.
