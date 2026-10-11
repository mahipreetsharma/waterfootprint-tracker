import os
from ultralytics import YOLO


def load_yolo_model():
  # Looks for model inside models/ directory, falls back to downloading/loading nano weights
  model_path = os.path.join(
      os.path.dirname(__file__), "..", "models", "yolov8n.pt"
  )
  if not os.path.exists(model_path):
    model_path = "yolov8n.pt"
  return YOLO(model_path)


def process_frame(frame, model, water_db):
  results = model(frame, verbose=False)
  annotated_frame = results[0].plot()

  detected_items = []
  for box in results[0].boxes:
    cls_id = int(box.cls[0])
    class_name = model.names[cls_id]

    if class_name == 'person':
        continue

    if class_name in water_db and class_name not in detected_items:
      detected_items.append(class_name)

  return annotated_frame, detected_items