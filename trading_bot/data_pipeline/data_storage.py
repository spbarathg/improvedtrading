import asyncio
import sqlite3
import pandas as pd
import logging
import numpy as np
from collections import deque
from datetime import datetime, timedelta

class DataStorage:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger()

        # In-memory storage for the most recent raw data and features
        self.recent_raw_data = deque(maxlen=config.memory_size)
        self.recent_features = deque(maxlen=config.memory_size)

        # SQLite connection for persistent storage
        self.db_conn = sqlite3.connect(config.db_path)
        self._initialize_db()

    def _initialize_db(self):
        """Create the necessary tables in the database if they don't exist."""
        with self.db_conn:
            self.db_conn.execute('''
                CREATE TABLE IF NOT EXISTS raw_data (
                    timestamp DATETIME,
                    data TEXT
                )
            ''')
            self.db_conn.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    timestamp DATETIME,
                    feature_data BLOB
                )
            ''')

    async def store_raw_data(self, source, raw_data):
        """Store raw data in memory and persist it."""
        try:
            timestamp = datetime.utcnow()

            # Store in-memory (latest data)
            self.recent_raw_data.append({
                'timestamp': timestamp,
                'source': source,
                'data': raw_data
            })

            # Store persistently in SQLite
            with self.db_conn:
                self.db_conn.execute(
                    "INSERT INTO raw_data (timestamp, data) VALUES (?, ?)",
                    (timestamp, str(raw_data))
                )

        except Exception as e:
            self.logger.error(f"Error storing raw data: {e}")

    async def store_features(self, features):
        """Store extracted features in memory and persist them."""
        try:
            timestamp = datetime.utcnow()

            # Store in-memory (latest features)
            self.recent_features.append({
                'timestamp': timestamp,
                'features': features
            })

            # Store persistently in SQLite as a BLOB
            with self.db_conn:
                self.db_conn.execute(
                    "INSERT INTO features (timestamp, feature_data) VALUES (?, ?)",
                    (timestamp, features.tobytes())
                )

        except Exception as e:
            self.logger.error(f"Error storing features: {e}")

    async def get_raw_data(self):
        """Retrieve the most recent raw data from memory."""
        return list(self.recent_raw_data)

    async def get_features(self):
        """Retrieve the most recent features from memory."""
        return list(self.recent_features)

    async def get_historical_data(self, days=7):
        """Retrieve a rolling window of historical raw data for the last 'days'."""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            query = "SELECT timestamp, data FROM raw_data WHERE timestamp >= ?"
            data = pd.read_sql_query(query, self.db_conn, params=[cutoff])

            # Convert to dictionary or appropriate format if needed
            return data

        except Exception as e:
            self.logger.error(f"Error retrieving historical data: {e}")
            return None

    async def clean_data(self, data):
        """Perform basic data cleaning (e.g., handling missing values)."""
        try:
            # Example: Remove rows with missing price or volume
            cleaned_data = [d for d in data if 'price' in d and 'volume' in d]
            return cleaned_data
        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            return data  # Return original if cleaning fails