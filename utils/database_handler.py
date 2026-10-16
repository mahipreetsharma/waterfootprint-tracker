from pymongo import MongoClient
from functools import lru_cache

# Connect to MongoDB
client = MongoClient(
    "mongodb://localhost:27017",
    serverSelectionTimeoutMS=5000
)

# Database and collection
db = client["waterFootprints"]
collection = db["waterFootprints"]


@lru_cache(maxsize=1)
def load_water_data():
    document = collection.find_one({}, {"_id": 0})

    if document is None:
        print("No data found in MongoDB!")
        return {}

    print("MongoDB data loaded successfully!")
    return document


def get_item_footprint(query):
    water_db = load_water_data()
    query = query.lower().strip()

    # Exact match
    if query in water_db:
        return water_db[query]

    # Common aliases
    aliases = {
        "bottle": "plastic bottle",
        "mobile": "cell phone",
        "phone": "cell phone",
        "computer": "desktop computer",
        "mouse": "computer mouse",
        "tv": "television",
        "tshirt": "cotton t-shirt",
        "t-shirt": "cotton t-shirt",
    }

    item_name = aliases.get(query)

    if item_name:
        return water_db.get(item_name)

    # Partial match
    for item in water_db:
        if query in item:
            return water_db[item]

    return None
import os
from pymongo import MongoClient
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Connect to MongoDB Atlas
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise ValueError(
        "MONGODB_URI not found! Please check your .env file."
    )

client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=10000
)

# Verify Atlas connection
client.admin.command("ping")
print("Connected to MongoDB Atlas successfully!")

# Database and collection
db = client["waterFootprintDB"]
collection = db["waterFootprints"]


@lru_cache(maxsize=1)
def load_water_data():
    document = collection.find_one({}, {"_id": 0})

    if document is None:
        print("No data found in MongoDB Atlas!")
        return {}

    print("MongoDB Atlas data loaded successfully!")
    return document


def get_item_footprint(query):
    water_db = load_water_data()
    query = query.lower().strip()

    # Exact match
    if query in water_db:
        return water_db[query]

    # Common aliases
    aliases = {
        "bottle": "plastic bottle",
        "mobile": "cell phone",
        "phone": "cell phone",
        "computer": "desktop computer",
        "mouse": "computer mouse",
        "tv": "television",
        "tshirt": "cotton t-shirt",
        "t-shirt": "cotton t-shirt",
    }

    item_name = aliases.get(query)

    if item_name:
        return water_db.get(item_name)

    # Partial match
    for item in water_db:
        if query in item:
            return water_db[item]

    return None
