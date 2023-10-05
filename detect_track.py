from ultralytics import YOLO
model = YOLO("yolov8x.pt")  # Loading model
result=model.track("datasets/mot20/1/1.mp4",save=True,save_txt=True)  # Predict