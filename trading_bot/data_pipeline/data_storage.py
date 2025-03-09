import asyncio
import logging
import orjson
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, AsyncGenerator, Type, TypeVar
from pydantic import BaseModel, ValidationError, validator, Field
from aiosqlite import Connection, Cursor, connect
from cachetools import TTLCache
from prometheus_client import Counter, Histogram, Gauge
from pathlib import Path
import zstandard as zstd
from trading_bot.config import config

logger = logging.getLogger(__name__)

# Metrics
STORAGE_OPS = Counter('storage_operations', 'Storage operations', ['operation', 'status'])
STORAGE_LATENCY = Histogram('storage_latency_seconds', 'Storage operation latency', ['operation'])
CACHE_EFFICIENCY = Gauge('storage_cache_efficiency', 'Cache hit ratio')
ACTIVE_CONNS = Gauge('storage_active_connections', 'Active database connections')

# Compression
_COMPRESSOR = zstd.ZstdCompressor(level=3)
_DECOMPRESSOR = zstd.ZstdDecompressor()

# Default configuration values
DEFAULT_FEATURE_VECTOR_SIZE = 100

T = TypeVar('T', bound='BaseModel')

class StorageModel(BaseModel):
    class Config:
        json_loads = orjson.loads
        json_dumps = lambda v, *, default: orjson.dumps(v, default=default).decode()

class RawData(StorageModel):
    category: str = Field(..., max_length=50)
    timestamp: datetime
    data: dict
    metadata: Optional[dict]

class MarketData(StorageModel):
    symbol: str = Field(..., max_length=20)
    timestamp: datetime
    price: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    metadata: Optional[dict]

class SocialMetrics(StorageModel):
    source: str = Field(..., max_length=20)
    timestamp: datetime
    metrics: dict
    metadata: Optional[dict]

class FeatureVector(StorageModel):
    timestamp: datetime
    values: List[float]
    metadata: Optional[dict]

    @validator('values')
    def validate_values(cls, v):
        if len(v) > getattr(config, 'FEATURE_VECTOR_SIZE', DEFAULT_FEATURE_VECTOR_SIZE):
            raise ValueError('Feature vector size exceeds limit')
        return v

class DataStorage:
    _SCHEMA = {
        'raw_data': RawData,
        'market_data': MarketData,
        'social_metrics': SocialMetrics,
        'features': FeatureVector
    }

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._pool: List[Connection] = []
        self._pool_lock = asyncio.Lock()
        self._max_pool_size = 10
        self._cache = TTLCache(maxsize=10_000, ttl=300)
        self._compression_enabled = True
        self._compression_threshold = 1024  # Only compress data larger than 1KB
        self._compression_level = 3
        self._max_batch_size = 1000  # Maximum number of items per batch
        self._prepared_statements: Dict[str, str] = {}
        self._last_connection_check = datetime.now()
        self._connection_check_interval = timedelta(minutes=5)
        self._query_timeout = 30  # seconds
        self._cache_stats = {'hits': 0, 'misses': 0}
        self._last_cache_cleanup = datetime.now()
        self._cache_cleanup_interval = timedelta(minutes=15)

    async def _check_connections(self):
        """Check and clean up stale connections"""
        now = datetime.now()
        if now - self._last_connection_check < self._connection_check_interval:
            return

        async with self._pool_lock:
            self._last_connection_check = now
            valid_connections = []
            for conn in self._pool:
                try:
                    # Simple health check
                    await conn.execute('SELECT 1')
                    valid_connections.append(conn)
                except Exception:
                    try:
                        await conn.close()
                    except Exception:
                        pass
            self._pool = valid_connections

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[Connection, None]:
        await self._check_connections()
        async with self._pool_lock:
            if self._pool:
                conn = self._pool.pop()
                try:
                    ACTIVE_CONNS.inc()
                    yield conn
                finally:
                    if len(self._pool) < self._max_pool_size:
                        self._pool.append(conn)
                    else:
                        await conn.close()
                    ACTIVE_CONNS.dec()
            else:
                async with self._connect() as conn:
                    ACTIVE_CONNS.inc()
                    yield conn

    @asynccontextmanager
    async def _connect(self) -> AsyncGenerator[Connection, None]:
        conn = await connect(
            self.db_path,
            timeout=10,
            isolation_level=None,
            cached_statements=100
        )
        try:
            await conn.execute('PRAGMA journal_mode=WAL')
            await conn.execute('PRAGMA synchronous=NORMAL')
            await conn.execute('PRAGMA cache_size=-10000')  # 10MB cache
            yield conn
        finally:
            await conn.close()
            ACTIVE_CONNS.dec()

    async def initialize(self):
        """Initialize database with optimized schema"""
        feature_vector_size = getattr(config, 'FEATURE_VECTOR_SIZE', DEFAULT_FEATURE_VECTOR_SIZE)
        async with self.connection() as conn:
            await conn.executescript(f'''
                CREATE TABLE IF NOT EXISTS raw_data (
                    id INTEGER PRIMARY KEY,
                    category TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    data BLOB NOT NULL,
                    metadata BLOB,
                    _compressed BOOLEAN DEFAULT 1
                ) STRICT;

                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    price REAL NOT NULL CHECK(price > 0),
                    volume REAL NOT NULL CHECK(volume >= 0),
                    metadata BLOB
                ) STRICT;

                CREATE TABLE IF NOT EXISTS social_metrics (
                    id INTEGER PRIMARY KEY,
                    source TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    metrics BLOB NOT NULL,
                    metadata BLOB
                ) STRICT;

                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    values BLOB NOT NULL,
                    metadata BLOB,
                    CHECK(json_array_length(values) <= {feature_vector_size})
                ) STRICT;

                -- Optimized indexes for common query patterns
                CREATE INDEX IF NOT EXISTS idx_raw_data_category_timestamp ON raw_data(category, timestamp);
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
                CREATE INDEX IF NOT EXISTS idx_social_source_timestamp ON social_metrics(source, timestamp);
                CREATE INDEX IF NOT EXISTS idx_features_timestamp ON features(timestamp);
                
                -- Partial indexes for frequently accessed recent data
                CREATE INDEX IF NOT EXISTS idx_recent_raw_data ON raw_data(timestamp) 
                    WHERE timestamp > datetime('now', '-1 day');
                CREATE INDEX IF NOT EXISTS idx_recent_market_data ON market_data(timestamp) 
                    WHERE timestamp > datetime('now', '-1 day');
                CREATE INDEX IF NOT EXISTS idx_recent_social_metrics ON social_metrics(timestamp) 
                    WHERE timestamp > datetime('now', '-1 day');
                CREATE INDEX IF NOT EXISTS idx_recent_features ON features(timestamp) 
                    WHERE timestamp > datetime('now', '-1 day');
            ''')

    def _compress(self, data: bytes) -> bytes:
        if not self._compression_enabled or len(data) < self._compression_threshold:
            return data
        return _COMPRESSOR.compress(data)

    def _decompress(self, data: bytes) -> bytes:
        if not self._compression_enabled or len(data) < self._compression_threshold:
            return data
        return _DECOMPRESSOR.decompress(data)

    def _get_cache_key(self, model: Type[T], data: dict) -> str:
        """Generate an efficient cache key"""
        # Use a faster hashing method for small data
        if len(data) < 100:
            return f"{model.__name__}:{hash(str(data))}"
        # For larger data, use a more efficient method
        return f"{model.__name__}:{hash(str(sorted(data.items())))}"

    async def _execute(self, conn: Connection, query: str, params: tuple) -> Cursor:
        start = datetime.now()
        try:
            # Set query timeout
            await conn.execute(f'PRAGMA busy_timeout = {self._query_timeout * 1000}')
            
            cursor = await conn.execute(query, params)
            STORAGE_LATENCY.labels(operation=query.split()[0]).observe(
                (datetime.now() - start).total_seconds()
            )
            return cursor
        except Exception as e:
            STORAGE_OPS.labels(operation=query.split()[0], status='error').inc()
            logger.error('Query failed: %s', e, exc_info=True)
            raise

    def _get_prepared_statement(self, table: str, columns: List[str]) -> str:
        """Get or create a prepared statement for a table"""
        key = f"{table}:{','.join(columns)}"
        if key not in self._prepared_statements:
            self._prepared_statements[key] = f'''
            INSERT INTO {table} ({', '.join(columns)})
            VALUES ({', '.join(['?']*len(columns))})
            '''
        return self._prepared_statements[key]

    async def store(self, model: Type[T], data: dict) -> bool:
        """Generic store method with validation and compression"""
        cache_key = self._get_cache_key(model, data)
        
        try:
            if cache_key in self._cache:
                CACHE_EFFICIENCY.inc()
                return True

            validated = model(**data)
            serialized = orjson.dumps(validated.dict())
            compressed = self._compress(serialized)

            async with self.connection() as conn:
                table = model.__name__.lower()
                columns = list(validated.dict().keys())
                query = self._get_prepared_statement(table, columns)
                
                await self._execute(
                    conn,
                    query,
                    tuple(compressed if k == 'data' else v 
                          for k, v in validated.dict().items())
                )
                await conn.commit()

            self._cache[cache_key] = True
            STORAGE_OPS.labels(operation='store', status='success').inc()
            return True
        except ValidationError as e:
            STORAGE_OPS.labels(operation='store', status='validation_error').inc()
            logger.warning('Validation failed: %s', e)
            return False

    async def batch_store(self, model: Type[T], items: List[dict]) -> int:
        """Bulk insert with transaction and optimized validation"""
        if not items:
            return 0

        total_processed = 0
        # Process items in chunks to avoid memory issues
        for chunk_start in range(0, len(items), self._max_batch_size):
            chunk = items[chunk_start:chunk_start + self._max_batch_size]
            
            try:
                async with self.connection() as conn:
                    await conn.execute('BEGIN')
                    table = model.__name__.lower()
                    columns = model.__fields__.keys()
                    
                    # Pre-validate all items
                    validated_items = []
                    for item in chunk:
                        try:
                            validated = model(**item)
                            serialized = orjson.dumps(validated.dict())
                            compressed = self._compress(serialized)
                            validated_items.append(
                                tuple(compressed if k == 'data' else v 
                                      for k, v in validated.dict().items())
                            )
                        except ValidationError as e:
                            logger.warning('Validation failed for item: %s', e)
                            continue

                    if not validated_items:
                        await conn.rollback()
                        continue

                    # Use a single bulk insert with prepared statement
                    placeholders = ','.join(['(' + ','.join(['?']*len(columns)) + ')']*len(validated_items))
                    query = f'''
                    INSERT INTO {table} ({', '.join(columns)})
                    VALUES {placeholders}
                    '''
                    
                    # Flatten the parameters for bulk insert
                    params = [param for item in validated_items for param in item]
                    await self._execute(conn, query, tuple(params))
                    
                    await conn.commit()
                    total_processed += len(validated_items)
                    STORAGE_OPS.labels(operation='batch_store', status='success').inc()
                    
            except Exception as e:
                await conn.rollback()
                STORAGE_OPS.labels(operation='batch_store', status='error').inc()
                logger.error('Batch store failed: %s', e)
                continue

        return total_processed

    def _update_cache_stats(self, hit: bool):
        """Update cache statistics"""
        if hit:
            self._cache_stats['hits'] += 1
        else:
            self._cache_stats['misses'] += 1
        
        # Calculate and update cache efficiency metric
        total = self._cache_stats['hits'] + self._cache_stats['misses']
        if total > 0:
            CACHE_EFFICIENCY.set(self._cache_stats['hits'] / total)

    async def _cleanup_cache(self):
        """Clean up and optimize cache"""
        now = datetime.now()
        if now - self._last_cache_cleanup < self._cache_cleanup_interval:
            return

        self._last_cache_cleanup = now
        # Clear old cache entries
        self._cache.clear()
        # Reset cache statistics
        self._cache_stats = {'hits': 0, 'misses': 0}

    async def retrieve(self, model: Type[T], **filters) -> List[T]:
        """Generic retrieve with cache and decompression"""
        cache_key = f"{model.__name__}:{hash(frozenset(filters.items()))}"
        
        if cache_key in self._cache:
            self._update_cache_stats(True)
            return self._cache[cache_key]

        self._update_cache_stats(False)
        try:
            async with self.connection() as conn:
                table = model.__name__.lower()
                where = ' AND '.join(f"{k} = ?" for k in filters)
                params = tuple(filters.values())
                
                query = f'SELECT * FROM {table}'
                if where:
                    query += f' WHERE {where}'
                
                cursor = await self._execute(conn, query, params)
                
                results = []
                async for row in cursor:
                    data = {k: self._decompress(v) if k == 'data' else v 
                           for k, v in zip(row.keys(), row)}
                    results.append(model(**orjson.loads(data)))
                
                self._cache[cache_key] = results
                STORAGE_OPS.labels(operation='retrieve', status='success').inc()
                return results
        except Exception as e:
            STORAGE_OPS.labels(operation='retrieve', status='error').inc()
            logger.error('Retrieve failed: %s', e)
            return []

    async def retrieve_paginated(self, model: Type[T], page: int = 1, page_size: int = 100, **filters) -> List[T]:
        """Retrieve data with pagination"""
        offset = (page - 1) * page_size
        cache_key = f"{model.__name__}:{hash(frozenset(filters.items()))}:{page}:{page_size}"
        
        if cache_key in self._cache:
            self._update_cache_stats(True)
            return self._cache[cache_key]

        self._update_cache_stats(False)
        try:
            async with self.connection() as conn:
                table = model.__name__.lower()
                where = ' AND '.join(f"{k} = ?" for k in filters)
                params = tuple(filters.values())
                
                query = f'SELECT * FROM {table}'
                if where:
                    query += f' WHERE {where}'
                query += f' LIMIT {page_size} OFFSET {offset}'
                
                cursor = await self._execute(conn, query, params)
                
                results = []
                async for row in cursor:
                    data = {k: self._decompress(v) if k == 'data' else v 
                           for k, v in zip(row.keys(), row)}
                    results.append(model(**orjson.loads(data)))
                
                self._cache[cache_key] = results
                STORAGE_OPS.labels(operation='retrieve_paginated', status='success').inc()
                return results
        except Exception as e:
            STORAGE_OPS.labels(operation='retrieve_paginated', status='error').inc()
            logger.error('Paginated retrieve failed: %s', e)
            return []

    async def maintain(self):
        """Maintenance tasks (vacuum, analyze, cache management)"""
        async with self.connection() as conn:
            await self._execute(conn, 'PRAGMA optimize', ())
            await self._execute(conn, 'VACUUM', ())
            await self._cleanup_cache()
            
            # Update database statistics
            await conn.execute('ANALYZE')
            
            # Rebuild indexes if needed
            await conn.execute('REINDEX')

    async def purge_old_data(self, retention: timedelta = timedelta(days=30)):
        """Efficient data purging with batch deletion"""
        cutoff = datetime.now() - retention
        async with self.connection() as conn:
            await self._execute(
                conn,
                'DELETE FROM raw_data WHERE timestamp < ? LIMIT 1000',
                (cutoff.isoformat(),)
            )
            await conn.commit()

    async def get_latest_batch(self, batch_size: int) -> List[Dict]:
        """Get latest batch of raw data for processing"""
        async with self.connection() as conn:
            cursor = await conn.execute('''
                SELECT category, timestamp, data, metadata
                FROM raw_data
                WHERE processed = 0
                ORDER BY timestamp ASC
                LIMIT ?
            ''', (batch_size,))
            
            rows = await cursor.fetchall()
            results = []
            
            for row in rows:
                try:
                    data = orjson.loads(self._decompress(row[2]) if row[2] else b'{}')
                    metadata = orjson.loads(row[3]) if row[3] else {}
                    results.append({
                        'category': row[0],
                        'timestamp': datetime.fromisoformat(row[1]),
                        'data': data,
                        'metadata': metadata
                    })
                except Exception as e:
                    logger.error(f"Error processing row: {e}")
                    continue
            
            # Mark processed rows
            if results:
                await conn.execute('''
                    UPDATE raw_data
                    SET processed = 1
                    WHERE category = ? AND timestamp = ?
                ''', [(r['category'], r['timestamp'].isoformat()) for r in results])
            
            return results

    async def batch_store_features(self, features: List[Dict]) -> int:
        """Store batch of feature vectors with compression"""
        if not features:
            return 0
            
        async with self.connection() as conn:
            # Prepare data for batch insert
            values = []
            for feature in features:
                try:
                    values.append((
                        feature['timestamp'].isoformat(),
                        orjson.dumps(feature['values']),
                        orjson.dumps(feature.get('metadata', {}))
                    ))
                except Exception as e:
                    logger.error(f"Error preparing feature: {e}")
                    continue
            
            if not values:
                return 0
            
            # Batch insert with prepared statement
            stmt = self._get_prepared_statement('features', ['timestamp', 'values', 'metadata'])
            cursor = await self._execute(conn, stmt, values)
            return cursor.rowcount