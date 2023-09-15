from ultralytics import YOLO

#Initialize YOLO with the Model Name
model = YOLO("best.pt")

##Predict Method Takes all the parameters of the Command Line Interface

model.predict(source="0", show=True, conf=0.15)
