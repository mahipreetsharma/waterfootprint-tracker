import os
from functools import lru_cache
from typing import Any, Optional
from dotenv import load_dotenv
from utils.item_names import ITEM_ALIASES, normalize_item_name

load_dotenv()

DEFAULT_WATER_DATA = {
    # Agricultural / Food items
    "coffee": {"total": 140.0, "blue": 7.0, "green": 130.0, "grey": 3.0, "unit": "cup (125 ml)"},
    "tea": {"total": 30.0, "blue": 2.0, "green": 25.0, "grey": 3.0, "unit": "cup (250 ml)"},
    "apple": {"total": 125.0, "blue": 18.0, "green": 95.0, "grey": 12.0, "unit": "apple"},
    "banana": {"total": 160.0, "blue": 20.0, "green": 130.0, "grey": 10.0, "unit": "banana"},
    "orange": {"total": 80.0, "blue": 16.0, "green": 56.0, "grey": 8.0, "unit": "orange"},
    "broccoli": {"total": 57.0, "blue": 12.0, "green": 38.0, "grey": 7.0, "unit": "portion (200g)"},
    "carrot": {"total": 33.0, "blue": 6.0, "green": 23.0, "grey": 4.0, "unit": "carrot (100g)"},
    "rice": {"total": 2500.0, "blue": 850.0, "green": 1450.0, "grey": 200.0, "unit": "kg"},
    "sandwich": {"total": 300.0, "blue": 45.0, "green": 210.0, "grey": 45.0, "unit": "item"},
    "hot dog": {"total": 600.0, "blue": 75.0, "green": 450.0, "grey": 75.0, "unit": "item"},
    "pizza": {"total": 1260.0, "blue": 180.0, "green": 950.0, "grey": 130.0, "unit": "item"},
    "donut": {"total": 180.0, "blue": 25.0, "green": 135.0, "grey": 20.0, "unit": "item"},
    "cake": {"total": 950.0, "blue": 140.0, "green": 710.0, "grey": 100.0, "unit": "item"},
    "bread": {"total": 1600.0, "blue": 240.0, "green": 1120.0, "grey": 240.0, "unit": "kg"},
    "milk": {"total": 1020.0, "blue": 80.0, "green": 870.0, "grey": 70.0, "unit": "liter"},
    "egg": {"total": 200.0, "blue": 18.0, "green": 160.0, "grey": 22.0, "unit": "egg (60g)"},
    "beef": {"total": 15415.0, "blue": 550.0, "green": 14415.0, "grey": 450.0, "unit": "kg"},
    "chicken": {"total": 4325.0, "blue": 315.0, "green": 3545.0, "grey": 465.0, "unit": "kg"},
    "chocolate": {"total": 17196.0, "blue": 1720.0, "green": 14200.0, "grey": 1276.0, "unit": "kg"},
    "beer": {"total": 75.0, "blue": 12.0, "green": 55.0, "grey": 8.0, "unit": "glass (250 ml)"},
    "wine": {"total": 110.0, "blue": 15.0, "green": 85.0, "grey": 10.0, "unit": "glass (125 ml)"},

    # Goods, Electronics & COCO Classes
    "plastic bottle": {"total": 4.0, "blue": 2.0, "green": 0.5, "grey": 1.5, "unit": "bottle (500 ml)"},
    "bottle": {"total": 4.0, "blue": 2.0, "green": 0.5, "grey": 1.5, "unit": "bottle (500 ml)"},
    "wine glass": {"total": 35.0, "blue": 12.0, "green": 5.0, "grey": 18.0, "unit": "glass"},
    "cup": {"total": 25.0, "blue": 8.0, "green": 3.0, "grey": 14.0, "unit": "ceramic cup"},
    "fork": {"total": 18.0, "blue": 6.0, "green": 2.0, "grey": 10.0, "unit": "stainless steel fork"},
    "knife": {"total": 22.0, "blue": 8.0, "green": 2.0, "grey": 12.0, "unit": "stainless steel knife"},
    "spoon": {"total": 16.0, "blue": 5.0, "green": 2.0, "grey": 9.0, "unit": "stainless steel spoon"},
    "bowl": {"total": 40.0, "blue": 14.0, "green": 6.0, "grey": 20.0, "unit": "ceramic bowl"},
    "cotton t-shirt": {"total": 2500.0, "blue": 540.0, "green": 1620.0, "grey": 340.0, "unit": "shirt (250g)"},
    "t-shirt": {"total": 2500.0, "blue": 540.0, "green": 1620.0, "grey": 340.0, "unit": "shirt (250g)"},
    "shirt": {"total": 2500.0, "blue": 540.0, "green": 1620.0, "grey": 340.0, "unit": "shirt (250g)"},
    "jeans": {"total": 8000.0, "blue": 1200.0, "green": 5600.0, "grey": 1200.0, "unit": "pair of denim jeans (800g)"},
    "shoes": {"total": 8000.0, "blue": 1100.0, "green": 5800.0, "grey": 1100.0, "unit": "pair of leather shoes"},
    "leather shoes": {"total": 8000.0, "blue": 1100.0, "green": 5800.0, "grey": 1100.0, "unit": "pair of leather shoes"},
    "cell phone": {"total": 12760.0, "blue": 3800.0, "green": 1400.0, "grey": 7560.0, "unit": "smartphone (140g)"},
    "phone": {"total": 12760.0, "blue": 3800.0, "green": 1400.0, "grey": 7560.0, "unit": "smartphone (140g)"},
    "smartphone": {"total": 12760.0, "blue": 3800.0, "green": 1400.0, "grey": 7560.0, "unit": "smartphone (140g)"},
    "laptop": {"total": 20000.0, "blue": 6200.0, "green": 2400.0, "grey": 11400.0, "unit": "laptop"},
    "desktop computer": {"total": 35000.0, "blue": 11000.0, "green": 4000.0, "grey": 20000.0, "unit": "desktop computer"},
    "computer": {"total": 35000.0, "blue": 11000.0, "green": 4000.0, "grey": 20000.0, "unit": "desktop computer"},
    "computer mouse": {"total": 450.0, "blue": 150.0, "green": 50.0, "grey": 250.0, "unit": "mouse"},
    "mouse": {"total": 450.0, "blue": 150.0, "green": 50.0, "grey": 250.0, "unit": "mouse"},
    "keyboard": {"total": 850.0, "blue": 280.0, "green": 90.0, "grey": 480.0, "unit": "keyboard"},
    "television": {"total": 28000.0, "blue": 8500.0, "green": 3500.0, "grey": 16000.0, "unit": "TV (55-inch)"},
    "tv": {"total": 28000.0, "blue": 8500.0, "green": 3500.0, "grey": 16000.0, "unit": "TV (55-inch)"},
    "remote": {"total": 320.0, "blue": 110.0, "green": 35.0, "grey": 175.0, "unit": "remote"},
    "backpack": {"total": 3200.0, "blue": 800.0, "green": 1600.0, "grey": 800.0, "unit": "backpack"},
    "umbrella": {"total": 950.0, "blue": 260.0, "green": 390.0, "grey": 300.0, "unit": "umbrella"},
    "handbag": {"total": 4500.0, "blue": 950.0, "green": 2600.0, "grey": 950.0, "unit": "handbag"},
    "tie": {"total": 650.0, "blue": 150.0, "green": 380.0, "grey": 120.0, "unit": "silk/cotton tie"},
    "suitcase": {"total": 5200.0, "blue": 1400.0, "green": 2200.0, "grey": 1600.0, "unit": "suitcase"},
    "book": {"total": 500.0, "blue": 120.0, "green": 300.0, "grey": 80.0, "unit": "paperback book (300 pages)"},
    "clock": {"total": 380.0, "blue": 120.0, "green": 40.0, "grey": 220.0, "unit": "clock"},
    "vase": {"total": 210.0, "blue": 70.0, "green": 20.0, "grey": 120.0, "unit": "ceramic vase"},
    "scissors": {"total": 85.0, "blue": 30.0, "green": 10.0, "grey": 45.0, "unit": "scissors"},
    "toothbrush": {"total": 15.0, "blue": 6.0, "green": 2.0, "grey": 7.0, "unit": "plastic toothbrush"},
    "chair": {"total": 1800.0, "blue": 450.0, "green": 1100.0, "grey": 250.0, "unit": "wooden/metal chair"},
    "couch": {"total": 12000.0, "blue": 2500.0, "green": 7500.0, "grey": 2000.0, "unit": "fabric sofa"},
    "potted plant": {"total": 45.0, "blue": 15.0, "green": 25.0, "grey": 5.0, "unit": "plant"},
    "bed": {"total": 16000.0, "blue": 3200.0, "green": 10500.0, "grey": 2300.0, "unit": "mattress and frame"},
    "dining table": {"total": 6500.0, "blue": 1200.0, "green": 4500.0, "grey": 800.0, "unit": "wooden dining table"},
    "toilet": {"total": 350.0, "blue": 120.0, "green": 30.0, "grey": 200.0, "unit": "ceramic toilet bowl"},
    "microwave": {"total": 3800.0, "blue": 1200.0, "green": 450.0, "grey": 2150.0, "unit": "microwave"},
    "oven": {"total": 7500.0, "blue": 2400.0, "green": 900.0, "grey": 4200.0, "unit": "electric oven"},
    "toaster": {"total": 1200.0, "blue": 380.0, "green": 140.0, "grey": 680.0, "unit": "2-slice toaster"},
    "sink": {"total": 850.0, "blue": 280.0, "green": 70.0, "grey": 500.0, "unit": "stainless steel sink"},
    "refrigerator": {"total": 14500.0, "blue": 4600.0, "green": 1700.0, "grey": 8200.0, "unit": "refrigerator"},
    "car": {"total": 400000.0, "blue": 120000.0, "green": 40000.0, "grey": 240000.0, "unit": "passenger car"},
    "bicycle": {"total": 4800.0, "blue": 1500.0, "green": 500.0, "grey": 2800.0, "unit": "aluminum bicycle"},
    "motorcycle": {"total": 45000.0, "blue": 14000.0, "green": 5000.0, "grey": 26000.0, "unit": "motorcycle"},
}


def _get_mongodb_client() -> Optional[Any]:
    uri = os.getenv("MONGODB_URI") or "mongodb://localhost:27017"
    try:
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        return client
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_water_data() -> dict[str, Any]:
    """Load water footprint data from MongoDB if available, otherwise use comprehensive fallback dataset."""
    client = _get_mongodb_client()
    if client is not None:
        try:
            db_name = "waterFootprintDB" if "mongodb.net" in (os.getenv("MONGODB_URI") or "") else "waterFootprints"
            db = client[db_name]
            collection = db["waterFootprints"]
            doc = collection.find_one({}, {"_id": 0})
            if doc and isinstance(doc, dict) and len(doc) > 0:
                # Merge with default data so all standard classes remain covered
                merged = {**DEFAULT_WATER_DATA, **doc}
                return merged
        except Exception:
            pass
        finally:
            try:
                client.close()
            except Exception:
                pass

    return DEFAULT_WATER_DATA.copy()


def get_item_footprint(query: str) -> Optional[dict[str, Any]]:
    """Retrieve water footprint metrics for a product by exact, alias, or partial match."""
    water_db = load_water_data()
    if not query:
        return None

    query = normalize_item_name(query)

    # Exact match
    if query in water_db:
        return water_db[query]

    # Alias match
    alias_target = ITEM_ALIASES.get(query)
    if alias_target and alias_target in water_db:
        return water_db[alias_target]

    # Reverse alias match
    for alias, target in ITEM_ALIASES.items():
        if normalize_item_name(target) == query and alias in water_db:
            return water_db[alias]

    # Partial match
    for key, value in water_db.items():
        if query in key or key in query:
            return value

    return None
