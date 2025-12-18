"""
Test MongoDB connection
"""
import sys
sys.path.append('.')

from utils.mongodb_handler import MongoDBHandler

def test_connection():
    print("Testing MongoDB connection...")
    print("-" * 50)
    
    # Create handler
    handler = MongoDBHandler()
    
    # Test connection
    print(f"Connection string: {handler.connection_string[:50]}...")
    print(f"Database name: {handler.database_name}")
    
    success = handler.connect()
    
    if success:
        print("✓ Connection successful!")
        print(f"✓ Connected to database: {handler.database_name}")
        print(f"✓ Collection: {handler.collection.name}")
        
        # Test basic operations
        print("\nTesting database operations...")
        
        # Count documents
        count = handler.collection.count_documents({})
        print(f"✓ Current documents in collection: {count}")
        
        # List all collections in database
        collections = handler.db.list_collection_names()
        print(f"✓ Available collections: {collections}")
        
        handler.close()
        print("\n✓ Connection closed successfully")
        return True
    else:
        print("✗ Connection failed!")
        return False

if __name__ == "__main__":
    test_connection()
