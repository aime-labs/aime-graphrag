"""
Adapter pool for managing shared LLM and embedding model instances.
"""
import asyncio
import weakref
from typing import Dict, Any, Optional, List
import logging
from adapters.llm_adapter import LLMAdapterFactory
from adapters.embedding_adapter import EmbeddingAdapterFactory


class PooledAdapterContext:
    """Context manager for using pooled adapters."""
    
    def __init__(self, pool, adapter_type: str, instance: Any, pool_key: str = "default", smart_cache=None):
        self.pool = pool
        self.adapter_type = adapter_type
        self.instance = instance
        self.pool_key = pool_key
        self.adapter = None
        self.smart_cache = smart_cache
    
    async def __aenter__(self):
        if self.adapter_type == "llm":
            self.adapter = await self.pool.get_llm_adapter(self.instance, self.pool_key, self.smart_cache)
        elif self.adapter_type == "embedding":
            self.adapter = await self.pool.get_embedding_adapter(self.instance, self.pool_key, self.smart_cache)
        else:
            raise ValueError(f"Unknown adapter type: {self.adapter_type}")
        return self.adapter
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.adapter:
            if self.adapter_type == "llm":
                await self.pool.return_llm_adapter(self.adapter, self.pool_key)
            elif self.adapter_type == "embedding":
                await self.pool.return_embedding_adapter(self.adapter, self.pool_key)


class AdapterPool:
    """Pool for managing shared adapter instances to improve performance."""
    
    def __init__(self, logger: logging.Logger, max_pool_size: int = 5):
        self.logger = logger
        self.max_pool_size = max_pool_size
        self._llm_pools: Dict[str, List[Any]] = {}
        self._embedding_pools: Dict[str, List[Any]] = {}
        self._pool_locks: Dict[str, asyncio.Lock] = {}
        self._usage_counts: Dict[str, int] = {}
        self._active_adapters = weakref.WeakSet()
        # CRITICAL-001 FIX: Per-adapter locks to prevent concurrent usage
        self._adapter_locks: Dict[int, asyncio.Lock] = {}
        # Store credentials for specific pool keys (e.g., "judge")
        self._credentials: Dict[str, Dict[str, str]] = {}
    
    def set_credentials(self, pool_key: str, user: str, api_key: str) -> None:
        """Store credentials for a specific pool key."""
        self._credentials[pool_key] = {'user': user, 'api_key': api_key}
        self.logger.debug(f"Stored credentials for pool: {pool_key}")
    
    async def get_llm_adapter(self, llm_instance: Any, pool_key: str = "default", smart_cache=None) -> Any:
        """Get an LLM adapter from the pool or create a new one."""
        lock = self._pool_locks.setdefault(pool_key, asyncio.Lock())
        
        async with lock:
            # Try to get from pool
            if pool_key in self._llm_pools and self._llm_pools[pool_key]:
                adapter = self._llm_pools[pool_key].pop()
                # CRITICAL-001 FIX: Acquire per-adapter lock to prevent concurrent usage
                adapter_id = id(adapter)
                if adapter_id not in self._adapter_locks:
                    self._adapter_locks[adapter_id] = asyncio.Lock()
                await self._adapter_locks[adapter_id].acquire()
                self._usage_counts[f"llm_{pool_key}"] = self._usage_counts.get(f"llm_{pool_key}", 0) + 1
                self._active_adapters.add(adapter)
                self.logger.debug(f"Reused LLM adapter from pool: {pool_key}")
                return adapter
            
            # Get credentials for this pool key if available
            creds = self._credentials.get(pool_key, {})
            user = creds.get('user')
            api_key = creds.get('api_key')
            
            # Create new adapter with credentials
            base_adapter = LLMAdapterFactory.create_adapter(
                llm_instance, self.logger, user=user, api_key=api_key
            )
            
            # Wrap with caching if smart_cache is provided
            if smart_cache:
                from core.smart_cache import CachedLLMAdapter
                adapter = CachedLLMAdapter(base_adapter, smart_cache, pool_key)
            else:
                adapter = base_adapter
            
            # CRITICAL-001 FIX: Create and acquire lock for new adapter
            adapter_id = id(adapter)
            if adapter_id not in self._adapter_locks:
                self._adapter_locks[adapter_id] = asyncio.Lock()
            await self._adapter_locks[adapter_id].acquire()
            self._usage_counts[f"llm_{pool_key}"] = self._usage_counts.get(f"llm_{pool_key}", 0) + 1
            self._active_adapters.add(adapter)
            self.logger.debug(f"Created new LLM adapter for pool: {pool_key}")
            return adapter
    
    async def get_embedding_adapter(self, embedding_instance: Any, pool_key: str = "default", smart_cache=None) -> Any:
        """Get an embedding adapter from the pool or create a new one."""
        lock = self._pool_locks.setdefault(pool_key, asyncio.Lock())
        
        async with lock:
            # Try to get from pool
            if pool_key in self._embedding_pools and self._embedding_pools[pool_key]:
                adapter = self._embedding_pools[pool_key].pop()
                # CRITICAL-001 FIX: Acquire per-adapter lock to prevent concurrent usage
                adapter_id = id(adapter)
                if adapter_id not in self._adapter_locks:
                    self._adapter_locks[adapter_id] = asyncio.Lock()
                await self._adapter_locks[adapter_id].acquire()
                self._usage_counts[f"embedding_{pool_key}"] = self._usage_counts.get(f"embedding_{pool_key}", 0) + 1
                self._active_adapters.add(adapter)
                self.logger.debug(f"Reused embedding adapter from pool: {pool_key}")
                return adapter
            
            # Create new adapter
            base_adapter = EmbeddingAdapterFactory.create_adapter(embedding_instance, self.logger)
            
            # Wrap with caching if smart_cache is provided
            if smart_cache:
                from core.smart_cache import CachedEmbeddingAdapter
                adapter = CachedEmbeddingAdapter(base_adapter, smart_cache, pool_key)
            else:
                adapter = base_adapter
            
            # CRITICAL-001 FIX: Create and acquire lock for new adapter
            adapter_id = id(adapter)
            if adapter_id not in self._adapter_locks:
                self._adapter_locks[adapter_id] = asyncio.Lock()
            await self._adapter_locks[adapter_id].acquire()
            self._usage_counts[f"embedding_{pool_key}"] = self._usage_counts.get(f"embedding_{pool_key}", 0) + 1
            self._active_adapters.add(adapter)
            self.logger.debug(f"Created new embedding adapter for pool: {pool_key}")
            return adapter
    
    async def return_llm_adapter(self, adapter: Any, pool_key: str = "default") -> None:
        """Return an LLM adapter to the pool."""
        # CRITICAL-001 FIX: Release per-adapter lock before returning to pool
        adapter_id = id(adapter)
        if adapter_id in self._adapter_locks and self._adapter_locks[adapter_id].locked():
            self._adapter_locks[adapter_id].release()
        
        lock = self._pool_locks.setdefault(pool_key, asyncio.Lock())
        
        async with lock:
            if pool_key not in self._llm_pools:
                self._llm_pools[pool_key] = []
            
            # Only return to pool if under max size
            if len(self._llm_pools[pool_key]) < self.max_pool_size:
                self._llm_pools[pool_key].append(adapter)
                self.logger.debug(f"Returned LLM adapter to pool: {pool_key}")
            else:
                self.logger.debug(f"LLM pool {pool_key} full, discarding adapter")
                # Clean up lock if adapter is being discarded
                if adapter_id in self._adapter_locks:
                    del self._adapter_locks[adapter_id]
    
    async def return_embedding_adapter(self, adapter: Any, pool_key: str = "default") -> None:
        """Return an embedding adapter to the pool."""
        # CRITICAL-001 FIX: Release per-adapter lock before returning to pool
        adapter_id = id(adapter)
        if adapter_id in self._adapter_locks and self._adapter_locks[adapter_id].locked():
            self._adapter_locks[adapter_id].release()
        
        lock = self._pool_locks.setdefault(pool_key, asyncio.Lock())
        
        async with lock:
            if pool_key not in self._embedding_pools:
                self._embedding_pools[pool_key] = []
            
            # Only return to pool if under max size
            if len(self._embedding_pools[pool_key]) < self.max_pool_size:
                self._embedding_pools[pool_key].append(adapter)
                self.logger.debug(f"Returned embedding adapter to pool: {pool_key}")
            else:
                self.logger.debug(f"Embedding pool {pool_key} full, discarding adapter")
                # Clean up lock if adapter is being discarded
                if adapter_id in self._adapter_locks:
                    del self._adapter_locks[adapter_id]
    
    def get_llm_adapter_context(self, llm_instance: Any, pool_key: str = "default", smart_cache=None):
        """Get a context manager for an LLM adapter."""
        return PooledAdapterContext(self, "llm", llm_instance, pool_key, smart_cache)
    
    def get_embedding_adapter_context(self, embedding_instance: Any, pool_key: str = "default", smart_cache=None):
        """Get a context manager for an embedding adapter."""
        return PooledAdapterContext(self, "embedding", embedding_instance, pool_key, smart_cache)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about pool usage."""
        stats = {
            'llm_pools': {key: len(adapters) for key, adapters in self._llm_pools.items()},
            'embedding_pools': {key: len(adapters) for key, adapters in self._embedding_pools.items()},
            'usage_counts': self._usage_counts.copy(),
            'active_adapters': len(self._active_adapters)
        }
        return stats
    
    async def cleanup(self) -> None:
        """Clean up all pools."""
        for lock in self._pool_locks.values():
            async with lock:
                pass  # Just acquire and release to ensure no ongoing operations
        
        self._llm_pools.clear()
        self._embedding_pools.clear()
        self._pool_locks.clear()
        self._usage_counts.clear()
        self.logger.info("Adapter pools cleaned up")
