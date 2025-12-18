"""
MongoDB connection test - final version
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("MongoDB Connection Test")
print("="*60)

conn_string = os.getenv('MONGODB_CONNECTION_STRING')
db_name = os.getenv('MONGODB_DATABASE', 'prop_book')

print(f"\nDatabase: {db_name}")
print(f"Connection string: {conn_string[:40]}...")

try:
    print("\nConnecting to MongoDB Atlas...")
    client = MongoClient(conn_string, serverSelectionTimeoutMS=5000)
    
    # Test connection
    client.admin.command('ping')
    print("[SUCCESS] Connected to MongoDB!")
    
    # Get database
    db = client[db_name]
    print(f"[SUCCESS] Accessing database: {db_name}")
    
    # List collections
    collections = db.list_collection_names()
    print(f"[INFO] Available collections: {collections}")
    
    # Get position_changes collection
    coll = db['position_changes']
    count = coll.count_documents({})
    print(f"[INFO] Documents in 'position_changes' collection: {count}")
    
    # Test write permission
    print("\n[TEST] Testing write permission...")
    test_doc = {
        'test': True,
        'message': 'Connection test successful'
    }
    result = coll.insert_one(test_doc)
    print(f"[SUCCESS] Write test successful! Document ID: {result.inserted_id}")
    
    # Delete test document
    coll.delete_one({'_id': result.inserted_id})
    print("[SUCCESS] Test document cleaned up")
    
    client.close()
    print("\n[SUCCESS] All tests passed! MongoDB is ready to use.")
    print("="*60)
    
except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {str(e)}")
    print("="*60)
