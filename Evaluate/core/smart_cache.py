"""
Intelligent caching system for LLM responses and embeddings.
"""
import hashlib
import asyncio
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict
import logging


class LRUCache:
    """LRU Cache with size and time-based expiration."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = OrderedDict()
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired."""
        async with self.lock:
            if key not in self.cache:
                return None
            
            # Check if expired
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.timestamps.move_to_end(key)
            return self.cache[key]
    
    async def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        async with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.timestamps.move_to_end(key)
            else:
                # Add new entry
                self.cache[key] = value
                self.timestamps[key] = time.time()
                
                # Evict oldest if over capacity
                while len(self.cache) > self.max_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_ratio': getattr(self, '_hits', 0) / max(getattr(self, '_total', 1), 1)
        }


class SmartCache:
    """Smart caching system for LLM responses and embeddings."""
    
    def __init__(self, logger: logging.Logger, max_cache_size: int = 1000, ttl_seconds: int = 3600):
        self.logger = logger
        self.llm_cache = LRUCache(max_cache_size, ttl_seconds)
        self.embedding_cache = LRUCache(max_cache_size // 2, ttl_seconds * 2)  # Embeddings cache longer
        self.metrics_cache = LRUCache(max_cache_size // 4, ttl_seconds)
        
        # Statistics
        self.llm_hits = 0
        self.llm_misses = 0
        self.embedding_hits = 0
        self.embedding_misses = 0
        self.metrics_hits = 0
        self.metrics_misses = 0
    
    def _create_cache_key(self, text: str, model_info: str = "", extra_params: Optional[Dict[str, Any]] = None) -> str:
        """Create a deterministic cache key for the given input."""
        # CRITICAL-006 FIX: Normalize numeric values to avoid floating point serialization issues
        normalized_params = {}
        if extra_params:
            for k, v in extra_params.items():
                if isinstance(v, float):
                    # Round to 10 decimal places for deterministic serialization
                    # This handles cases like 0.7000000000000001 vs 0.7
                    normalized_params[k] = round(v, 10)
                elif isinstance(v, (int, str, bool, type(None))):
                    normalized_params[k] = v
                else:
                    # For complex types, convert to string representation
                    normalized_params[k] = str(v)
        
        # Create a hash of the text content and normalized parameters
        content = {
            'text': text.strip(),
            'model': model_info,
            'params': normalized_params
        }
        # CRITICAL-006 FIX: Use sort_keys=True for deterministic JSON serialization
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]  # Use first 16 chars
    
    async def get_llm_response(self, prompt: str, model_info: str = "", 
                             extra_params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get cached LLM response if available."""
        cache_key = self._create_cache_key(prompt, model_info, extra_params)
        result = await self.llm_cache.get(cache_key)
        
        if result is not None:
            self.llm_hits += 1
            self.logger.debug(f"LLM cache hit for key: {cache_key[:8]}")
            return result
        else:
            self.llm_misses += 1
            return None
    
    async def cache_llm_response(self, prompt: str, response: str, model_info: str = "",
                               extra_params: Optional[Dict[str, Any]] = None) -> None:
        """Cache LLM response."""
        cache_key = self._create_cache_key(prompt, model_info, extra_params)
        await self.llm_cache.put(cache_key, response)
        self.logger.debug(f"Cached LLM response for key: {cache_key[:8]}")
    
    async def get_embedding(self, text: str, model_info: str = "") -> Optional[List[float]]:
        """Get cached embedding if available."""
        cache_key = self._create_cache_key(text, model_info)
        result = await self.embedding_cache.get(cache_key)
        
        if result is not None:
            self.embedding_hits += 1
            self.logger.debug(f"Embedding cache hit for key: {cache_key[:8]}")
            return result
        else:
            self.embedding_misses += 1
            return None
    
    async def cache_embedding(self, text: str, embedding: List[float], model_info: str = "") -> None:
        """Cache embedding."""
        cache_key = self._create_cache_key(text, model_info)
        await self.embedding_cache.put(cache_key, embedding)
        self.logger.debug(f"Cached embedding for key: {cache_key[:8]}")
    
    async def get_metric_result(self, metric_name: str, inputs: Dict[str, Any]) -> Optional[float]:
        """Get cached metric result if available."""
        cache_key = self._create_cache_key(f"{metric_name}:{json.dumps(inputs, sort_keys=True)}")
        result = await self.metrics_cache.get(cache_key)
        
        if result is not None:
            self.metrics_hits += 1
            self.logger.debug(f"Metrics cache hit for {metric_name}")
            return result
        else:
            self.metrics_misses += 1
            return None
    
    async def cache_metric_result(self, metric_name: str, inputs: Dict[str, Any], result: float) -> None:
        """Cache metric result."""
        cache_key = self._create_cache_key(f"{metric_name}:{json.dumps(inputs, sort_keys=True)}")
        await self.metrics_cache.put(cache_key, result)
        self.logger.debug(f"Cached metric result for {metric_name}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_llm = self.llm_hits + self.llm_misses
        total_embedding = self.embedding_hits + self.embedding_misses
        total_metrics = self.metrics_hits + self.metrics_misses
        
        return {
            'llm_cache': {
                'hits': self.llm_hits,
                'misses': self.llm_misses,
                'hit_rate': self.llm_hits / max(total_llm, 1),
                'size': len(self.llm_cache.cache)
            },
            'embedding_cache': {
                'hits': self.embedding_hits,
                'misses': self.embedding_misses,
                'hit_rate': self.embedding_hits / max(total_embedding, 1),
                'size': len(self.embedding_cache.cache)
            },
            'metrics_cache': {
                'hits': self.metrics_hits,
                'misses': self.metrics_misses,
                'hit_rate': self.metrics_hits / max(total_metrics, 1),
                'size': len(self.metrics_cache.cache)
            }
        }
    
    async def clear_all_caches(self) -> None:
        """Clear all caches."""
        await self.llm_cache.clear()
        await self.embedding_cache.clear()
        await self.metrics_cache.clear()
        self.logger.info("All caches cleared")


class CachedLLMAdapter:
    """Wrapper around LLM adapter with caching."""
    
    def __init__(self, llm_adapter: Any, cache: SmartCache, model_info: str = ""):
        self.llm_adapter = llm_adapter
        self.cache = cache
        self.model_info = model_info
    
    async def ainvoke(self, prompt: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Invoke LLM with caching.
        
        Args:
            prompt: The prompt to send to the LLM
            config: Optional configuration dict (for compatibility with various adapters)
            **kwargs: Additional keyword arguments
        """
        # Merge config into kwargs if provided
        if config:
            kwargs['config'] = config
        
        # Check cache first
        cached_response = await self.cache.get_llm_response(prompt, self.model_info, kwargs)
        if cached_response is not None:
            # Return mock response object with cached content
            class MockResponse:
                def __init__(self, content):
                    self.content = content
            return MockResponse(cached_response)
        
        # Call actual LLM - pass config explicitly if provided for adapter compatibility
        if config:
            response = await self.llm_adapter.ainvoke(prompt, config=config)
        else:
            response = await self.llm_adapter.ainvoke(prompt)
        
        # Cache the response
        await self.cache.cache_llm_response(prompt, response.content, self.model_info, kwargs)
        
        return response


class CachedEmbeddingAdapter:
    """Wrapper around embedding adapter with caching."""
    
    def __init__(self, embedding_adapter: Any, cache: SmartCache, model_info: str = ""):
        self.embedding_adapter = embedding_adapter
        self.cache = cache
        self.model_info = model_info
    
    async def aembed_query(self, text: str, **kwargs) -> List[float]:
        """Get embedding with caching."""
        # Check cache first
        cached_embedding = await self.cache.get_embedding(text, self.model_info)
        if cached_embedding is not None:
            return cached_embedding
        
        # Call actual embedding model
        embedding = await self.embedding_adapter.aembed_query(text, **kwargs)
        
        # Cache the embedding
        await self.cache.cache_embedding(text, embedding, self.model_info)
        
        return embedding
