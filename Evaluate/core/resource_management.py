"""
Resource management and cleanup utilities.
"""

import asyncio
import aiohttp
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from contextlib import asynccontextmanager
import weakref
import gc
import psutil
import os

from interfaces.core_interfaces import ResourceManagerInterface


class TemporaryFileManager:
    """Manages temporary files with automatic cleanup."""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(tempfile.gettempdir())
        self.temp_files: Set[Path] = set()
        self.temp_dirs: Set[Path] = set()
        self.logger = logging.getLogger(__name__)
    
    def create_temp_file(self, suffix: str = "", prefix: str = "eval_", 
                        content: str = "") -> Path:
        """Create a temporary file."""
        temp_file = Path(tempfile.mktemp(suffix=suffix, prefix=prefix, dir=self.base_dir))
        
        try:
            with open(temp_file, 'w') as f:
                f.write(content)
            
            self.temp_files.add(temp_file)
            self.logger.debug(f"Created temporary file: {temp_file}")
            return temp_file
            
        except Exception as e:
            self.logger.error(f"Failed to create temporary file: {str(e)}")
            raise
    
    def create_temp_dir(self, prefix: str = "eval_") -> Path:
        """Create a temporary directory."""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=self.base_dir))
        self.temp_dirs.add(temp_dir)
        self.logger.debug(f"Created temporary directory: {temp_dir}")
        return temp_dir
    
    def cleanup_file(self, file_path: Path) -> None:
        """Clean up a specific temporary file."""
        try:
            if file_path.exists():
                file_path.unlink()
                self.temp_files.discard(file_path)
                self.logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up file {file_path}: {str(e)}")
    
    def cleanup_dir(self, dir_path: Path) -> None:
        """Clean up a specific temporary directory."""
        try:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                self.temp_dirs.discard(dir_path)
                self.logger.debug(f"Cleaned up temporary directory: {dir_path}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up directory {dir_path}: {str(e)}")
    
    def cleanup_all(self) -> None:
        """Clean up all temporary files and directories."""
        # Clean up files
        for file_path in self.temp_files.copy():
            self.cleanup_file(file_path)
        
        # Clean up directories
        for dir_path in self.temp_dirs.copy():
            self.cleanup_dir(dir_path)
        
        self.logger.info("Cleaned up all temporary files and directories")


class ConnectionPoolManager:
    """Manages HTTP connection pools for API calls."""
    
    def __init__(self, pool_size: int = 10, timeout: int = 30):
        self.pool_size = pool_size
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with connection pooling."""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.pool_size,
                limit_per_host=self.pool_size // 2,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            
            self.logger.info(f"Created HTTP session with pool size {self.pool_size}")
        
        return self.session
    
    async def close(self) -> None:
        """Close the HTTP session and cleanup connections."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("Closed HTTP session")


class MemoryManager:
    """Monitors and manages memory usage."""
    
    def __init__(self, max_memory_mb: Optional[int] = None):
        self.max_memory_mb = max_memory_mb
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': memory_percent
        }
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds limit."""
        if self.max_memory_mb is None:
            return True
        
        usage = self.get_memory_usage()
        if usage['rss_mb'] > self.max_memory_mb:
            self.logger.warning(f"Memory usage ({usage['rss_mb']:.1f} MB) exceeds limit ({self.max_memory_mb} MB)")
            return False
        
        return True
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return collected objects."""
        before = self.get_memory_usage()
        
        # Force garbage collection
        collected = gc.collect()
        
        after = self.get_memory_usage()
        
        self.logger.info(f"Garbage collection: {collected} objects collected, "
                        f"memory reduced by {before['rss_mb'] - after['rss_mb']:.1f} MB")
        
        return {
            'objects_collected': collected,
            'memory_freed_mb': int(before['rss_mb'] - after['rss_mb'])
        }


class EvaluationResourceManager(ResourceManagerInterface):
    """Comprehensive resource manager for evaluation framework."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize managers
        self.temp_file_manager = TemporaryFileManager(
            config.get('temp_dir')
        )
        
        self.connection_pool = ConnectionPoolManager(
            pool_size=config.get('connection_pool_size', 10),
            timeout=config.get('request_timeout', 30)
        )
        
        max_memory_str = config.get('max_memory_usage')
        max_memory_mb = None
        if max_memory_str:
            # Parse memory string like "8GB" to MB
            if max_memory_str.upper().endswith('GB'):
                max_memory_mb = float(max_memory_str[:-2]) * 1024
            elif max_memory_str.upper().endswith('MB'):
                max_memory_mb = float(max_memory_str[:-2])
        
        self.memory_manager = MemoryManager(int(max_memory_mb) if max_memory_mb else None)
        
        # Track resource usage
        self.cleanup_enabled = config.get('cleanup_temp_files', True)
        self._closed = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.logger.info("Initializing evaluation resources")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.cleanup()
    
    def create_temp_file(self, suffix: str = "", prefix: str = "eval_", 
                        content: str = "") -> Path:
        """Create a temporary file."""
        return self.temp_file_manager.create_temp_file(suffix, prefix, content)
    
    def create_temp_dir(self, prefix: str = "eval_") -> Path:
        """Create a temporary directory."""
        return self.temp_file_manager.create_temp_dir(prefix)
    
    async def get_http_session(self) -> aiohttp.ClientSession:
        """Get HTTP session for API calls."""
        return await self.connection_pool.get_session()
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        return self.memory_manager.check_memory_limit()
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        memory_usage = self.memory_manager.get_memory_usage()
        
        return {
            'memory_usage': memory_usage,
            'temp_files_count': len(self.temp_file_manager.temp_files),
            'temp_dirs_count': len(self.temp_file_manager.temp_dirs),
            'http_session_active': self.connection_pool.session is not None and not self.connection_pool.session.closed
        }
    
    async def cleanup(self) -> None:
        """Cleanup all managed resources."""
        if self._closed:
            return
        
        self.logger.info("Starting resource cleanup")
        
        try:
            # Close HTTP connections
            await self.connection_pool.close()
            
            # Clean up temporary files if enabled
            if self.cleanup_enabled:
                self.temp_file_manager.cleanup_all()
            
            # Force garbage collection
            self.memory_manager.force_garbage_collection()
            
            self._closed = True
            self.logger.info("Resource cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {str(e)}")


@asynccontextmanager
async def managed_resources(config: Dict[str, Any]):
    """Context manager for automatic resource management."""
    resource_manager = EvaluationResourceManager(config)
    
    try:
        async with resource_manager:
            yield resource_manager
    finally:
        # Ensure cleanup even if there's an exception
        if not resource_manager._closed:
            await resource_manager.cleanup()


class ResourceMonitor:
    """Monitors resource usage during evaluation."""
    
    def __init__(self, resource_manager: EvaluationResourceManager, 
                 check_interval: int = 60):
        self.resource_manager = resource_manager
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = False
    
    async def start_monitoring(self):
        """Start resource monitoring."""
        self._stop_monitoring = False
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Started resource monitoring")
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self._stop_monitoring = True
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped resource monitoring")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring:
            try:
                # Check memory usage
                if not self.resource_manager.check_memory_usage():
                    # Force garbage collection if memory is high
                    self.resource_manager.memory_manager.force_garbage_collection()
                
                # Log resource stats
                stats = self.resource_manager.get_resource_stats()
                self.logger.debug(f"Resource stats: {stats}")
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {str(e)}")
                await asyncio.sleep(self.check_interval)
