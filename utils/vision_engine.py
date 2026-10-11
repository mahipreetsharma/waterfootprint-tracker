import os
from ultralytics import YOLO


def load_yolo_model():
  # Points to model/yolo26n.pt relative to the utils folder
  model_path = os.path.join(
      os.path.dirname(__file__), "..", "model", "yolo26n.pt"
  )

  if not os.path.exists(model_path):
    # Fallback to default download if the file isn't found
    model_path = "yolo26n.pt"

  return YOLO(model_path)


def process_frame(frame, model, water_db):
  # Automatically find class IDs in the YOLO model that match items in your water_db
  allowed_classes = [
      cls_id for cls_id, name in model.names.items() if name.lower() in water_db
  ]

  # Run YOLO inference, restricting detection ONLY to items in your database
  results = model(frame, classes=allowed_classes, verbose=False)
  annotated_frame = results[0].plot()

  detected_items = []
  for box in results[0].boxes:
    cls_id = int(box.cls[0])
    class_name = model.names[cls_id]
    if class_name in water_db and class_name not in detected_items:
      detected_items.append(class_name)

  return annotated_frame, detected_items