from dotenv import load_dotenv
import os
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)

db = client["multimodal_db"]

users_collection = db["users"]
analysis_collection = db["analysis_results"]
