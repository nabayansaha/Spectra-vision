from ultralytics import YOLO
model = YOLO("best.pt")
results = model.predict(source="output/spots_aug2.jpg", save=True, conf=0.25)  
print(results)
