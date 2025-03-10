import os
import shutil
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from trading_bot.utils.decorators import error_handler
import aiosqlite
import orjson
import zstandard as zstd
from cachetools import TTLCache

logger = logging.getLogger(__name__)

class DataManager:
    """Manages data versioning, retention, and cleanup."""
    
    def __init__(self, config):
        self.config = config
        self.data_dir = Path(config.DATA_STORAGE_PATH).parent
        self.version_dir = self.data_dir / 'versions'
        self.metadata_file = self.version_dir / 'metadata.json'
        self.retention_days = getattr(config, 'DATA_RETENTION_DAYS', 30)
        self.max_versions = getattr(config, 'MAX_DATA_VERSIONS', 5)
        self.db_path = config.DATA_STORAGE_PATH
        self._cache = TTLCache(maxsize=10_000, ttl=300)  # 5 minutes cache
        self._compressor = zstd.ZstdCompressor(level=3)
        self._decompressor = zstd.ZstdDecompressor()
        
        # Create necessary directories
        self.version_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load metadata
        self._metadata = self._load_metadata()
        
        logger.info("Initializing data manager with path: %s", self.db_path)
        
    def _load_metadata(self) -> Dict:
        """Load version metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'versions': [], 'last_cleanup': None}
        
    def _save_metadata(self):
        """Save version metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2, default=str)
            
    @error_handler("create_version")
    def create_version(self, description: str) -> str:
        """Create a new version of the data."""
        timestamp = datetime.now().isoformat()
        version_id = f"v_{timestamp.replace(':', '-')}"
        version_path = self.version_dir / version_id
        
        # Create version directory
        version_path.mkdir(exist_ok=True)
        
        # Copy current data to version directory
        if self.config.DATA_STORAGE_PATH.exists():
            shutil.copy2(
                self.config.DATA_STORAGE_PATH,
                version_path / self.config.DATA_STORAGE_PATH.name
            )
            
        # Update metadata
        version_info = {
            'id': version_id,
            'timestamp': timestamp,
            'description': description,
            'size': self._get_directory_size(version_path)
        }
        self._metadata['versions'].append(version_info)
        self._save_metadata()
        
        # Cleanup old versions
        self._cleanup_old_versions()
        
        return version_id
        
    @error_handler("restore_version")
    def restore_version(self, version_id: str) -> bool:
        """Restore data from a specific version."""
        version_path = self.version_dir / version_id
        if not version_path.exists():
            raise ValueError(f"Version {version_id} not found")
            
        # Backup current state before restore
        self.create_version("Auto-backup before restore")
        
        # Restore data
        shutil.copy2(
            version_path / self.config.DATA_STORAGE_PATH.name,
            self.config.DATA_STORAGE_PATH
        )
        
        logger.info(f"Restored data from version {version_id}")
        return True
        
    def _cleanup_old_versions(self):
        """Clean up old versions based on retention policy."""
        now = datetime.now()
        retention_date = now - timedelta(days=self.retention_days)
        
        # Sort versions by timestamp
        versions = sorted(
            self._metadata['versions'],
            key=lambda x: datetime.fromisoformat(x['timestamp'])
        )
        
        # Keep only the maximum number of versions
        if len(versions) > self.max_versions:
            versions_to_remove = versions[:-self.max_versions]
            for version in versions_to_remove:
                self._remove_version(version['id'])
                
        # Remove versions older than retention period
        for version in versions:
            version_date = datetime.fromisoformat(version['timestamp'])
            if version_date < retention_date:
                self._remove_version(version['id'])
                
        self._metadata['last_cleanup'] = now.isoformat()
        self._save_metadata()
        
    def _remove_version(self, version_id: str):
        """Remove a specific version."""
        version_path = self.version_dir / version_id
        if version_path.exists():
            shutil.rmtree(version_path)
            
        self._metadata['versions'] = [
            v for v in self._metadata['versions']
            if v['id'] != version_id
        ]
        
    def _get_directory_size(self, path: Path) -> int:
        """Calculate total size of a directory in bytes."""
        total = 0
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
        return total
        
    def get_versions(self) -> List[Dict]:
        """Get list of all versions."""
        return self._metadata['versions']
        
    def get_version_info(self, version_id: str) -> Optional[Dict]:
        """Get information about a specific version."""
        for version in self._metadata['versions']:
            if version['id'] == version_id:
                return version
        return None

    async def initialize(self):
        """Initialize database schema"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        data BLOB NOT NULL,
                        metadata BLOB
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time 
                    ON market_data(symbol, timestamp DESC);
                ''')
                await db.commit()
                logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize database schema: %s", str(e), exc_info=True)
            raise

    async def store_data(self, symbol: str, data: Dict):
        """Store market data with compression"""
        try:
            compressed_data = self._compressor.compress(orjson.dumps(data))
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    'INSERT INTO market_data (symbol, timestamp, data) VALUES (?, ?, ?)',
                    (symbol, datetime.now().isoformat(), compressed_data)
                )
                await db.commit()
                logger.debug("Stored data for symbol: %s", symbol)
        except Exception as e:
            logger.error("Failed to store data for %s: %s", symbol, str(e), exc_info=True)
            raise

    async def get_data(self, symbol: str, start_time: Optional[datetime] = None) -> List[Dict]:
        """Retrieve market data with caching"""
        try:
            cache_key = f"{symbol}:{start_time.isoformat() if start_time else 'latest'}"
            if cache_key in self._cache:
                logger.debug("Cache hit for %s", cache_key)
                return self._cache[cache_key]

            query = 'SELECT data FROM market_data WHERE symbol = ?'
            params = [symbol]
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time.isoformat())
            
            query += ' ORDER BY timestamp DESC'
            
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    data = [
                        orjson.loads(self._decompressor.decompress(row[0]))
                        for row in rows
                    ]
                    self._cache[cache_key] = data
                    logger.debug("Retrieved %d records for %s", len(data), symbol)
                    return data
        except Exception as e:
            logger.error("Failed to retrieve data for %s: %s", symbol, str(e), exc_info=True)
            return []

    async def cleanup_old_data(self, days: int = 7):
        """Remove data older than specified days"""
        try:
            cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    'DELETE FROM market_data WHERE timestamp < ?',
                    (cutoff_time,)
                )
                deleted = cursor.rowcount
                await db.commit()
                logger.info("Cleaned up %d old records", deleted)
        except Exception as e:
            logger.error("Failed to cleanup old data: %s", str(e), exc_info=True)
            raise

    def clear_cache(self):
        """Clear the data cache"""
        try:
            self._cache.clear()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error("Failed to clear cache: %s", str(e), exc_info=True) 