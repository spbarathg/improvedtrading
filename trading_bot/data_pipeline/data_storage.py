import asyncio
import logging
from datetime import datetime, timedelta
import json
import aioredis
import asyncpg
from typing import List, Dict, Optional, Any
import numpy as np

class DataStorage:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger()
        self.redis_pool = None
        self.pg_pool = None
        
    async def initialize(self):
        """Initialize Redis and PostgreSQL connections."""
        try:
            # Initialize Redis connection pool
            self.redis_pool = await aioredis.create_redis_pool(
                f'redis://{self.config.redis_host}:{self.config.redis_port}',
                password=self.config.redis_password,
                minsize=5,
                maxsize=20
            )

            # Initialize PostgreSQL connection pool
            self.pg_pool = await asyncpg.create_pool(
                user=self.config.pg_user,
                password=self.config.pg_password,
                database=self.config.pg_database,
                host=self.config.pg_host,
                port=self.config.pg_port,
                min_size=5,
                max_size=20
            )

            # Initialize database tables
            await self._initialize_db()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize storage connections: {e}")
            raise

    async def _initialize_db(self):
        """Create necessary PostgreSQL tables if they don't exist."""
        async with self.pg_pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS raw_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE,
                    source VARCHAR(255),
                    data JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE,
                    feature_data JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for better query performance
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_raw_data_timestamp ON raw_data(timestamp)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_features_timestamp ON features(timestamp)')

    async def store_raw_data(self, source: str, raw_data: Dict[str, Any]) -> None:
        """Store raw data in Redis and PostgreSQL."""
        try:
            timestamp = datetime.utcnow()
            data_key = f"raw_data:{timestamp.isoformat()}"
            
            # Store in Redis with expiration (e.g., 24 hours)
            data_json = json.dumps({
                'timestamp': timestamp.isoformat(),
                'source': source,
                'data': raw_data
            })
            await self.redis_pool.setex(
                data_key,
                self.config.redis_expiry_seconds,
                data_json
            )

            # Store in PostgreSQL for persistence
            async with self.pg_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO raw_data (timestamp, source, data)
                    VALUES ($1, $2, $3)
                ''', timestamp, source, json.dumps(raw_data))

        except Exception as e:
            self.logger.error(f"Error storing raw data: {e}")
            raise

    async def store_features(self, features: Dict[str, Any]) -> None:
        """Store extracted features in Redis and PostgreSQL."""
        try:
            timestamp = datetime.utcnow()
            feature_key = f"features:{timestamp.isoformat()}"
            
            # Store in Redis with expiration
            feature_json = json.dumps({
                'timestamp': timestamp.isoformat(),
                'features': features
            })
            await self.redis_pool.setex(
                feature_key,
                self.config.redis_expiry_seconds,
                feature_json
            )

            # Store in PostgreSQL
            async with self.pg_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO features (timestamp, feature_data)
                    VALUES ($1, $2)
                ''', timestamp, json.dumps(features))

        except Exception as e:
            self.logger.error(f"Error storing features: {e}")
            raise

    async def get_raw_data(self, limit: int = None) -> List[Dict[str, Any]]:
        """Retrieve the most recent raw data from Redis."""
        try:
            # Get all keys matching the pattern
            keys = await self.redis_pool.keys('raw_data:*')
            if not keys:
                return []

            # Sort keys by timestamp (newest first) and limit if specified
            sorted_keys = sorted(keys, reverse=True)
            if limit:
                sorted_keys = sorted_keys[:limit]

            # Get data for all keys
            data = []
            for key in sorted_keys:
                raw_json = await self.redis_pool.get(key)
                if raw_json:
                    data.append(json.loads(raw_json))
            
            return data

        except Exception as e:
            self.logger.error(f"Error retrieving raw data: {e}")
            return []

    async def get_features(self, limit: int = None) -> List[Dict[str, Any]]:
        """Retrieve the most recent features from Redis."""
        try:
            # Get all keys matching the pattern
            keys = await self.redis_pool.keys('features:*')
            if not keys:
                return []

            # Sort keys by timestamp (newest first) and limit if specified
            sorted_keys = sorted(keys, reverse=True)
            if limit:
                sorted_keys = sorted_keys[:limit]

            # Get features for all keys
            features = []
            for key in sorted_keys:
                feature_json = await self.redis_pool.get(key)
                if feature_json:
                    features.append(json.loads(feature_json))
            
            return features

        except Exception as e:
            self.logger.error(f"Error retrieving features: {e}")
            return []

    async def get_historical_data(self, days: int = 7) -> List[Dict[str, Any]]:
        """Retrieve historical raw data from PostgreSQL."""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT timestamp, source, data
                    FROM raw_data
                    WHERE timestamp >= $1
                    ORDER BY timestamp DESC
                ''', cutoff)
                
                return [
                    {
                        'timestamp': row['timestamp'],
                        'source': row['source'],
                        'data': row['data']
                    }
                    for row in rows
                ]

        except Exception as e:
            self.logger.error(f"Error retrieving historical data: {e}")
            return []

    async def clean_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform asynchronous data cleaning operations."""
        try:
            # Basic cleaning operations
            cleaned_data = [
                d for d in data
                if all(key in d.get('data', {}) for key in ['price', 'volume', 'liquidity'])
            ]

            # Additional async cleaning operations can be added here
            return cleaned_data

        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            return data

    async def close(self):
        """Close all database connections properly."""
        try:
            if self.redis_pool:
                self.redis_pool.close()
                await self.redis_pool.wait_closed()
            
            if self.pg_pool:
                await self.pg_pool.close()
                
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")
            raise

    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data from PostgreSQL periodically."""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days_to_keep)
            
            async with self.pg_pool.acquire() as conn:
                # Delete old raw data
                await conn.execute('''
                    DELETE FROM raw_data
                    WHERE timestamp < $1
                ''', cutoff)
                
                # Delete old features
                await conn.execute('''
                    DELETE FROM features
                    WHERE timestamp < $1
                ''', cutoff)

        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            raise