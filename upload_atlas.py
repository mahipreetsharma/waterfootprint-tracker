import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

uri = os.getenv("MONGODB_URI")

if not uri:
    raise ValueError("MONGODB_URI .env file mein nahi mili!")

client = MongoClient(uri, serverSelectionTimeoutMS=15000)

try:
    # Check Atlas connection
    client.admin.command("ping")
    print("MongoDB Atlas connected!")

    # Read local JSON dataset
    with open("data/water_data.json", "r", encoding="utf-8") as file:
        water_data = json.load(file)

    db = client["waterFootprintDB"]
    collection = db["waterFootprints"]

    print("Products in JSON file:", len(water_data))
    print("Existing documents in Atlas:", collection.count_documents({}))

    if collection.count_documents({}) == 0:
        collection.insert_one(water_data)
        print("SUCCESS: Water footprint data uploaded to Atlas!")

    else:
        print("Collection already contains documents.")
        print("No data was inserted to avoid duplicates.")

    # Verify uploaded data
    document = collection.find_one({}, {"_id": 0})

    if document:
        print("Keys found in Atlas:", list(document.keys())[:10])

finally:
    client.close()