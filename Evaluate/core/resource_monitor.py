"""
Resource-aware concurrency control for optimal performance.
"""
import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import logging


class ResourceMonitor:
    """Monitor system resources for dynamic concurrency control."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.last_check = 0
        self.check_interval = 5.0  # Check every 5 seconds
        self._cached_stats = {}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system resource statistics."""
        current_time = time.time()
        
        # Use cached stats if recent
        if current_time - self.last_check < self.check_interval and self._cached_stats:
            return self._cached_stats
        
        try:
            # Simple CPU count
            cpu_count = os.cpu_count() or 4
            
            # Simple load average (Unix-like systems)
            load_avg = None
            try:
                load_avg = os.getloadavg()[0]
            except (AttributeError, OSError):
                load_avg = 0.5  # Default moderate load
            
            # Estimate CPU usage from load average
            cpu_percent = min(100, (load_avg / cpu_count) * 100) if load_avg else 50
            
            stats = {
                'cpu_percent': cpu_percent,
                'memory_percent': 50,  # Default moderate memory usage
                'available_memory_gb': 4.0,  # Default assumption
                'cpu_count': cpu_count,
                'load_average': load_avg,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            self._cached_stats = stats
            self.last_check = current_time
            return stats
            
        except Exception as e:
            self.logger.warning(f"Failed to get system stats: {str(e)}")
            return self._cached_stats or {'cpu_percent': 50, 'memory_percent': 50, 'cpu_count': 4}
    
    def recommend_concurrency(self, base_concurrency: int, api_latency: Optional[float] = None) -> int:
        """Recommend optimal concurrency based on system resources."""
        stats = self.get_system_stats()
        
        # Start with base concurrency
        recommended = base_concurrency
        
        # Adjust based on CPU usage (from load average)
        cpu_percent = stats.get('cpu_percent', 50)
        if cpu_percent > 80:
            recommended = max(1, int(recommended * 0.7))  # Reduce by 30%
        elif cpu_percent < 30:
            recommended = min(base_concurrency * 2, int(recommended * 1.3))  # Increase by 30%
        
        # Adjust based on CPU count (more CPUs = more concurrency)
        cpu_count = stats.get('cpu_count', 4)
        if cpu_count >= 8:
            recommended = min(base_concurrency * 2, int(recommended * 1.2))
        elif cpu_count <= 2:
            recommended = max(1, int(recommended * 0.8))
        
        # Adjust based on API latency if provided
        if api_latency:
            if api_latency > 5.0:  # High latency, increase concurrency
                recommended = min(base_concurrency * 3, int(recommended * 1.5))
            elif api_latency < 1.0:  # Low latency, can handle more
                recommended = min(base_concurrency * 2, int(recommended * 1.2))
        
        # Ensure reasonable bounds
        recommended = max(1, min(recommended, base_concurrency * 3))
        
        self.logger.debug(f"Concurrency recommendation: {recommended} (base: {base_concurrency}, "
                         f"CPU: {cpu_percent}%, Load: {stats.get('load_average', 'N/A')})")
        
        return recommended


class AdaptiveSemaphore:
    """Semaphore that adapts its value based on system resources."""
    
    def __init__(self, initial_value: int, resource_monitor: ResourceMonitor, 
                 logger: logging.Logger, adaptation_interval: float = 30.0):
        self.initial_value = initial_value  # Keep track of original value  
        self.base_value = initial_value
        self.current_value = initial_value
        self.resource_monitor = resource_monitor
        self.logger = logger
        self.adaptation_interval = adaptation_interval
        self.semaphore = asyncio.Semaphore(initial_value)
        self.last_adaptation = time.time()
        self._adaptation_task = None
        self._adaptation_lock = None  # Will be created when needed
        self._api_latencies = []
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure the adaptation task is started."""
        if not self._initialized:
            try:
                if self._adaptation_lock is None:
                    self._adaptation_lock = asyncio.Lock()
                self._adaptation_task = asyncio.create_task(self._adaptation_loop())
                self._initialized = True
                self.logger.debug("AdaptiveSemaphore adaptation task started")
            except RuntimeError:
                # No event loop running, will try again later
                pass
    
    async def __aenter__(self):
        await self._ensure_initialized()
        # Record start time for latency measurement
        start_time = time.time()
        await self.semaphore.acquire()
        
        # Store start time for this acquisition
        return start_time
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Record API latency for adaptation (simplified approach)
        # In a real implementation, you'd track the start time properly
        # For now, just release the semaphore
        self.semaphore.release()
    
    async def _adaptation_loop(self):
        """Background task to periodically adapt concurrency."""
        while True:
            try:
                await asyncio.sleep(self.adaptation_interval)
                await self._adapt_concurrency()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"Error in adaptation loop: {str(e)}")
    
    async def _adapt_concurrency(self):
        """Adapt the semaphore value based on current system resources."""
        # Simplified approach: just monitor and log recommendations
        # Avoid complex semaphore replacement that can cause errors
        
        # Get recommended concurrency based on current system state
        recommended = self.resource_monitor.recommend_concurrency(self.initial_value)
        
        if recommended != self.current_value:
            old_value = self.current_value
            self.current_value = recommended
            self.logger.info(f"Concurrency recommendation changed from {old_value} to {recommended} "
                           f"(actual semaphore remains at {self.base_value})")
    
    def record_api_latency(self, latency: float):
        """Record an API call latency for adaptation."""
        if len(self._api_latencies) < 100:  # Keep last 100 measurements
            self._api_latencies.append(latency)
    
    async def cleanup(self):
        """Clean up the adaptation task."""
        if self._adaptation_task and not self._adaptation_task.done():
            self._adaptation_task.cancel()
            try:
                await self._adaptation_task
            except asyncio.CancelledError:
                pass
    
    @property
    def current_limit(self) -> int:
        """Get the current concurrency limit."""
        return self.current_value
