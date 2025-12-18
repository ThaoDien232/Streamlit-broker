"""
Local CSV-based storage for position changes
Simple and fast - no database needed
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
import os

class PositionChangesHandler:
    """Handler for position changes using local CSV storage"""
    
    def __init__(self, csv_path="sql/position_changes.csv"):
        """Initialize handler with CSV file path"""
        self.csv_path = csv_path
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=['broker', 'quarter', 'news_info', 'update_date'])
            df.to_csv(self.csv_path, index=False)
    
    def add_position_change(self, broker, quarter, news_info, update_date=None):
        """
        Add a position change record
        
        Args:
            broker: Broker name (VIX, VCI, HCM, SHS, etc.)
            quarter: Quarter (e.g., "4Q25")
            news_info: News/information text
            update_date: Update date (defaults to now)
        
        Returns:
            bool: True if successful
        """
        try:
            # Load existing data
            df = pd.read_csv(self.csv_path)
            
            # Create new record
            new_record = pd.DataFrame([{
                'broker': broker.upper(),
                'quarter': quarter,
                'news_info': news_info,
                'update_date': update_date or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }])
            
            # Append and save
            df = pd.concat([df, new_record], ignore_index=True)
            df.to_csv(self.csv_path, index=False)
            
            return True
        except Exception as e:
            print(f"Error adding position change: {e}")
            return False
    
    def get_position_changes(self, broker=None, quarter=None):
        """
        Retrieve position changes based on filters
        
        Args:
            broker: Filter by broker code (optional)
            quarter: Filter by quarter (optional)
        
        Returns:
            List of dictionaries with position change data
        """
        try:
            df = pd.read_csv(self.csv_path)
            
            # Apply filters
            if broker:
                df = df[df['broker'] == broker.upper()]
            if quarter:
                df = df[df['quarter'] == quarter]
            
            # Sort by update_date descending (newest first)
            df = df.sort_values('update_date', ascending=False)
            
            # Convert to list of dictionaries
            return df.to_dict('records')
        except Exception as e:
            print(f"Error retrieving position changes: {e}")
            return []
    
    def delete_position_change(self, index):
        """
        Delete a position change record by index
        
        Args:
            index: Row index to delete
        
        Returns:
            bool: True if successful
        """
        try:
            df = pd.read_csv(self.csv_path)
            df = df.drop(index)
            df.to_csv(self.csv_path, index=False)
            return True
        except Exception as e:
            print(f"Error deleting position change: {e}")
            return False
