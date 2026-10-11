import json
import os


def load_water_data():
  path = os.path.join(
      os.path.dirname(__file__), "..", "data", "water_data.json"
  )
  try:
    with open(path, "r") as f:
      return json.load(f)
  except FileNotFoundError:
    return {}


def get_item_footprint(query):
  water_db = load_water_data()
  query = query.lower().strip()
  return water_db.get(query, None)