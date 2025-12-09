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
    
    def __init__(self, pool, adapter_type: str, instance: Any, pool_key: str = "default"):
        self.pool = pool
        self.adapter_type = adapter_type
        self.instance = instance
        self.pool_key = pool_key
        self.adapter = None
    
    async def __aenter__(self):
        if self.adapter_type == "llm":
            self.adapter = await self.pool.get_llm_adapter(self.instance, self.pool_key)
        elif self.adapter_type == "embedding":
            self.adapter = await self.pool.get_embedding_adapter(self.instance, self.pool_key)
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
    
    async def get_llm_adapter(self, llm_instance: Any, pool_key: str = "default") -> Any:
        """Get an LLM adapter from the pool or create a new one."""
        lock = self._pool_locks.setdefault(pool_key, asyncio.Lock())
        
        async with lock:
            # Try to get from pool
            if pool_key in self._llm_pools and self._llm_pools[pool_key]:
                adapter = self._llm_pools[pool_key].pop()
                self._usage_counts[f"llm_{pool_key}"] = self._usage_counts.get(f"llm_{pool_key}", 0) + 1
                self._active_adapters.add(adapter)
                self.logger.debug(f"Reused LLM adapter from pool: {pool_key}")
                return adapter
            
            # Create new adapter
            adapter = LLMAdapterFactory.create_adapter(llm_instance, self.logger)
            self._usage_counts[f"llm_{pool_key}"] = self._usage_counts.get(f"llm_{pool_key}", 0) + 1
            self._active_adapters.add(adapter)
            self.logger.debug(f"Created new LLM adapter for pool: {pool_key}")
            return adapter
    
    async def get_embedding_adapter(self, embedding_instance: Any, pool_key: str = "default") -> Any:
        """Get an embedding adapter from the pool or create a new one."""
        lock = self._pool_locks.setdefault(pool_key, asyncio.Lock())
        
        async with lock:
            # Try to get from pool
            if pool_key in self._embedding_pools and self._embedding_pools[pool_key]:
                adapter = self._embedding_pools[pool_key].pop()
                self._usage_counts[f"embedding_{pool_key}"] = self._usage_counts.get(f"embedding_{pool_key}", 0) + 1
                self._active_adapters.add(adapter)
                self.logger.debug(f"Reused embedding adapter from pool: {pool_key}")
                return adapter
            
            # Create new adapter
            adapter = EmbeddingAdapterFactory.create_adapter(embedding_instance, self.logger)
            self._usage_counts[f"embedding_{pool_key}"] = self._usage_counts.get(f"embedding_{pool_key}", 0) + 1
            self._active_adapters.add(adapter)
            self.logger.debug(f"Created new embedding adapter for pool: {pool_key}")
            return adapter
    
    async def return_llm_adapter(self, adapter: Any, pool_key: str = "default") -> None:
        """Return an LLM adapter to the pool."""
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
    
    async def return_embedding_adapter(self, adapter: Any, pool_key: str = "default") -> None:
        """Return an embedding adapter to the pool."""
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
    
    def get_llm_adapter_context(self, llm_instance: Any, pool_key: str = "default"):
        """Get a context manager for an LLM adapter."""
        return PooledAdapterContext(self, "llm", llm_instance, pool_key)
    
    def get_embedding_adapter_context(self, embedding_instance: Any, pool_key: str = "default"):
        """Get a context manager for an embedding adapter."""
        return PooledAdapterContext(self, "embedding", embedding_instance, pool_key)
    
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
