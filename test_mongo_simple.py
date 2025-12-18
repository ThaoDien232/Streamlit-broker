"""
Simple MongoDB connection test
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Loading environment variables...")
conn_string = os.getenv('MONGODB_CONNECTION_STRING')
db_name = os.getenv('MONGODB_DATABASE', 'prop_book')

print(f"Connection string loaded: {conn_string[:30]}...")
print(f"Database name: {db_name}")

print("\nAttempting to connect...")
try:
    client = MongoClient(conn_string, serverSelectionTimeoutMS=5000)
    
    # Test connection
    client.admin.command('ping')
    print("✓ Connection successful!")
    
    # Get database
    db = client[db_name]
    print(f"✓ Connected to database: {db_name}")
    
    # List collections
    collections = db.list_collection_names()
    print(f"✓ Available collections: {collections}")
    
    # Get position_changes collection
    coll = db['position_changes']
    count = coll.count_documents({})
    print(f"✓ Documents in 'position_changes': {count}")
    
    client.close()
    print("\n✓ Test completed successfully!")
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}")
    print(f"✗ Message: {str(e)}")
