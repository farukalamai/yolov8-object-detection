from ultralytics import YOLO

#Initialize YOLO with the Model Name
model = YOLO("yolo_model/yolov8n.pt")

##Predict Method Takes all the parameters of the Command Line Interface

model.predict(source='data/demo.mp4', save=True, conf=0.5, save_txt=True)
model.export(format="onnx")