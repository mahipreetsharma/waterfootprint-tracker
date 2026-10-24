import os
from collections.abc import Mapping
from copy import copy

from ultralytics import YOLO

from utils.item_names import ITEM_ALIASES, PERSON_NAMES, normalize_item_name


def load_yolo_model():
  # Points to model/yolo26x.pt relative to the utils folder
  model_path = os.path.join(
      os.path.dirname(__file__), "..", "model", "yolo26n.pt"
  )

  if not os.path.exists(model_path):
    # Fallback to default download if the file isn't found
    model_path = "yolo26n.pt"

  return YOLO(model_path)


def _model_class_names(model):
    names = model.names
    if isinstance(names, Mapping):
        return ((int(class_id), name) for class_id, name in names.items())
    return enumerate(names)


def _database_item_index(water_db):
    items = {}
    for item in water_db:
        if not isinstance(item, str):
            continue
        normalized = normalize_item_name(item)
        if normalized and normalized not in PERSON_NAMES:
            items.setdefault(normalized, item)
    return items


def _resolve_database_item(class_name, database_items):
    normalized_class = normalize_item_name(class_name)
    if normalized_class in PERSON_NAMES:
        return None

    candidates = [normalized_class]
    alias_target = ITEM_ALIASES.get(normalized_class)
    if alias_target:
        candidates.append(normalize_item_name(alias_target))

    candidates.extend(
        alias
        for alias, target in ITEM_ALIASES.items()
        if normalize_item_name(target) == normalized_class
    )

    for candidate in candidates:
        if candidate not in PERSON_NAMES and candidate in database_items:
            return database_items[candidate]
    return None


def process_frame(frame, model, water_db):
    """Detect only model-supported database items, excluding people."""
    database_items = _database_item_index(water_db)
    class_to_item = {}
    supported_names = []

    for class_id, class_name in _model_class_names(model):
        if not isinstance(class_name, str):
            continue
        normalized_name = normalize_item_name(class_name)
        if normalized_name in PERSON_NAMES:
            continue

        supported_names.append(normalized_name)
        database_item = _resolve_database_item(class_name, database_items)
        if database_item is not None:
            class_to_item[class_id] = database_item

    if not class_to_item:
        if not supported_names:
            raise ValueError(
                "This YOLO model exposes no non-person classes for detection."
            )
        labels = ", ".join(supported_names)
        raise ValueError(
            "None of this model's non-person classes match items in the water "
            "database. The model can detect only its supported labels, not "
            "arbitrary database products. Supported labels: "
            f"{labels}"
        )

    # Restrict inference to mapped classes; person classes are never eligible.
    results = model(frame, classes=list(class_to_item), verbose=False)
    result = results[0]
    detected_items = []
    allowed_box_indices = []
    for index, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        database_item = class_to_item.get(class_id)
        if database_item is None:
            continue
        allowed_box_indices.append(index)
        if database_item not in detected_items:
            detected_items.append(database_item)

    if len(allowed_box_indices) == len(result.boxes):
        annotated_frame = result.plot()
    else:
        filtered_result = copy(result)
        boxes = result.boxes
        if isinstance(boxes, list):
            filtered_result.boxes = [boxes[index] for index in allowed_box_indices]
        else:
            filtered_result.boxes = boxes[allowed_box_indices]
        annotated_frame = filtered_result.plot()

    return annotated_frame, detected_items
