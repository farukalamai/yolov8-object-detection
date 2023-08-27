from ultralytics import YOLO

#Initialize YOLO with the Model Name
model = YOLO("yolo_model/yolov8n-seg.pt")

##Predict Method Takes all the parameters of the Command Line Interface

model.predict(source='data/yolo_basics/demo.mp4', save=True, conf=0.8, save_txt=True,)

# which confidence value less than 0.8 that will not show up

model.export(format="onnx")