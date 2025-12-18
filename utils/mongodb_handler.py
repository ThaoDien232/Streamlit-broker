"""
MongoDB handler for prop book position changes tracking
"""
import os
from datetime import datetime
from typing import List, Dict, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MongoDBHandler:
    """Handler for MongoDB operations related to position changes"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        self.connection_string = os.getenv('MONGODB_CONNECTION_STRING')
        self.database_name = os.getenv('MONGODB_DATABASE', 'prop_book')
        self.client = None
        self.db = None
        self.collection = None
        
    def connect(self):
        """Establish connection to MongoDB"""
        try:
            self.client = MongoClient(self.connection_string)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            self.collection = self.db['position_changes']
            return True
        except ConnectionFailure as e:
            print(f"Failed to connect to MongoDB: {e}")
            return False
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            return False
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
    
    def add_position_change(self,
                          broker: str,
                          quarter: str,
                          news_info: str,
                          update_date: Optional[datetime] = None) -> bool:
        """
        Add a position change record to MongoDB

        Simple format with 4 fields:
        - broker: Broker name (VIX, VCI, HCM)
        - quarter: Quarter (e.g., "1Q25")
        - news_info: News/information text
        - update_date: Update date (defaults to now)

        Returns:
            bool: True if successful, False otherwise
        """
        if self.collection is None:
            if not self.connect():
                return False

        try:
            document = {
                'broker': broker.upper(),
                'quarter': quarter,
                'news_info': news_info,
                'update_date': update_date or datetime.now()
            }

            result = self.collection.insert_one(document)
            return result.inserted_id is not None
        except OperationFailure as e:
            print(f"Failed to insert document: {e}")
            return False
        except Exception as e:
            print(f"Error inserting document: {e}")
            return False
    
    def get_position_changes(self,
                           broker: Optional[str] = None,
                           quarter: Optional[str] = None) -> List[Dict]:
        """
        Retrieve position changes based on filters

        Args:
            broker: Filter by broker code
            quarter: Filter by quarter

        Returns:
            List of position change documents
        """
        if self.collection is None:
            if not self.connect():
                return []

        try:
            query = {}
            if broker:
                query['broker'] = broker.upper()
            if quarter:
                query['quarter'] = quarter

            results = list(self.collection.find(query).sort('update_date', -1))
            return results
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def update_position_change(self,
                             document_id: str,
                             updates: Dict) -> bool:
        """
        Update an existing position change document

        Args:
            document_id: MongoDB document _id
            updates: Dictionary of fields to update

        Returns:
            bool: True if successful, False otherwise
        """
        if self.collection is None:
            if not self.connect():
                return False

        try:
            from bson.objectid import ObjectId

            result = self.collection.update_one(
                {'_id': ObjectId(document_id)},
                {'$set': updates}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating document: {e}")
            return False
    
    def delete_position_change(self, document_id: str) -> bool:
        """
        Delete a position change document
        
        Args:
            document_id: MongoDB document _id
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.collection is None:
            if not self.connect():
                return False

        try:
            from bson.objectid import ObjectId
            result = self.collection.delete_one({'_id': ObjectId(document_id)})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def get_all_quarters(self, broker: Optional[str] = None) -> List[str]:
        """
        Get all unique quarters in the database
        
        Args:
            broker: Optional broker filter
        
        Returns:
            List of quarter strings
        """
        if self.collection is None:
            if not self.connect():
                return []

        try:
            query = {}
            if broker:
                query['broker'] = broker.upper()

            quarters = self.collection.distinct('quarter', query)
            return sorted(quarters)
        except Exception as e:
            print(f"Error retrieving quarters: {e}")
            return []
