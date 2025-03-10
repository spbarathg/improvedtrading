import os
import shutil
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from trading_bot.utils.decorators import error_handler

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
        
        # Create necessary directories
        self.version_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load metadata
        self._metadata = self._load_metadata()
        
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