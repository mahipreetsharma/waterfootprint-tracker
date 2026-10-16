import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

uri = os.getenv("MONGODB_URI")

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    print(client.admin.command("ping"))
    print("MongoDB Atlas connected successfully!")
    client.close()
except Exception as e:
    print("Connection error:", e)