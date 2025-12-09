"""
Core interfaces for the evaluation framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import asyncio


class LLMInterface(ABC):
    """Abstract interface for Language Model adapters."""
    
    @abstractmethod
    async def ainvoke(self, prompt: str, **kwargs) -> Any:
        """Asynchronously invoke the LLM with a prompt."""
        pass
    
    @abstractmethod
    async def abatch(self, prompts: List[str], **kwargs) -> List[Any]:
        """Asynchronously process a batch of prompts."""
        pass


class EmbeddingInterface(ABC):
    """Abstract interface for Embedding model adapters."""
    
    @abstractmethod
    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed a single query."""
        pass
    
    @abstractmethod
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed multiple documents."""
        pass


class QueryRunnerInterface(ABC):
    """Abstract interface for query execution systems."""
    
    @abstractmethod
    async def run_query(self, question: str, context: Any = None, 
                       prompt_template: Optional[str] = None) -> Tuple[str, str, List[str]]:
        """Execute a query and return (raw_answer, processed_answer, contexts)."""
        pass
    
    @abstractmethod
    def get_supported_architectures(self) -> List[str]:
        """Return list of supported architectures."""
        pass


class MetricInterface(ABC):
    """Abstract interface for evaluation metrics."""
    
    @abstractmethod
    async def compute(self, **kwargs) -> float:
        """Compute the metric value."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""
        pass
    
    @property
    @abstractmethod
    def requires_llm(self) -> bool:
        """Return True if metric requires LLM for computation."""
        pass
    
    @property
    @abstractmethod
    def requires_embeddings(self) -> bool:
        """Return True if metric requires embeddings for computation."""
        pass


class ConfigurationInterface(ABC):
    """Abstract interface for configuration management."""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        pass
    
    @abstractmethod
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        pass


class DataLoaderInterface(ABC):
    """Abstract interface for data loading."""
    
    @abstractmethod
    async def load_questions(self, path: str) -> List[Dict[str, Any]]:
        """Load questions from file."""
        pass
    
    @abstractmethod
    async def validate_dataset(self, data: List[Dict[str, Any]]) -> List[str]:
        """Validate dataset and return list of errors."""
        pass


class ResourceManagerInterface(ABC):
    """Abstract interface for resource management."""
    
    @abstractmethod
    async def __aenter__(self):
        """Async context manager entry."""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Explicitly cleanup resources."""
        pass


class StatisticalAnalyzerInterface(ABC):
    """Abstract interface for statistical analysis."""
    
    @abstractmethod
    def compute_confidence_interval(self, scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for scores."""
        pass
    
    @abstractmethod
    def significance_test(self, scores1: List[float], scores2: List[float]) -> Dict[str, Any]:
        """Perform statistical significance test between two sets of scores."""
        pass
    
    @abstractmethod
    def bootstrap_analysis(self, scores: List[float], n_bootstrap: int = 1000) -> Dict[str, Any]:
        """Perform bootstrap analysis on scores."""
        pass
