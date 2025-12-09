from abc import ABC, abstractmethod
from typing import Any, List, Optional
import asyncio
import logging
import numpy as np


class EmbeddingAdapter(ABC):
    """Abstract base class for embedding adapters."""
    
    @abstractmethod
    async def aembed_query(self, text: str) -> List[float]:
        """Embed text asynchronously."""
        pass
    
    @abstractmethod
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents asynchronously."""
        pass


class HuggingFaceEmbeddingAdapter(EmbeddingAdapter):
    """Adapter for HuggingFace embeddings."""
    
    def __init__(self, embedding_model: Any, logger: logging.Logger):
        self.embedding_model = embedding_model
        self.logger = logger
    
    async def aembed_query(self, text: str) -> List[float]:
        """Embed single text query."""
        try:
            # Handle different embedding model interfaces
            if hasattr(self.embedding_model, 'aembed'):
                embedding = await self.embedding_model.aembed(text)
            elif hasattr(self.embedding_model, 'embed'):
                embedding = self.embedding_model.embed(text)
            else:
                raise AttributeError("Embedding model has no embed method")
            
            return embedding
        except Exception as e:
            self.logger.error(f"Embedding failed: {str(e)}")
            raise
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        try:
            embeddings = []
            for text in texts:
                embedding = await self.aembed_query(text)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            self.logger.error(f"Batch embedding failed: {str(e)}")
            raise


class OpenAIEmbeddingAdapter(EmbeddingAdapter):
    """Adapter for OpenAI embeddings."""
    
    def __init__(self, embedding_model: Any, logger: logging.Logger):
        self.embedding_model = embedding_model
        self.logger = logger
    
    async def aembed_query(self, text: str) -> List[float]:
        """Embed single text query."""
        try:
            if hasattr(self.embedding_model, 'aembed_query'):
                embedding = await self.embedding_model.aembed_query(text)
            elif hasattr(self.embedding_model, 'embed_query'):
                embedding = self.embedding_model.embed_query(text)
            else:
                raise AttributeError("OpenAI embedding model has no embed_query method")
            
            return embedding
        except Exception as e:
            self.logger.error(f"OpenAI embedding failed: {str(e)}")
            raise
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        try:
            embeddings = []
            for text in texts:
                embedding = await self.aembed_query(text)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            self.logger.error(f"OpenAI batch embedding failed: {str(e)}")
            raise


class LangChainEmbeddingAdapter(EmbeddingAdapter):
    """Adapter for LangChain embeddings."""
    
    def __init__(self, embedding_model: Any, logger: logging.Logger):
        self.embedding_model = embedding_model
        self.logger = logger
    
    async def aembed_query(self, text: str) -> List[float]:
        """Embed single text query."""
        try:
            if hasattr(self.embedding_model, 'aembed_query'):
                embedding = await self.embedding_model.aembed_query(text)
            elif hasattr(self.embedding_model, 'embed_query'):
                embedding = self.embedding_model.embed_query(text)
            else:
                raise AttributeError("LangChain embedding model has no embed_query method")
            
            return embedding
        except Exception as e:
            self.logger.error(f"LangChain embedding failed: {str(e)}")
            raise
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        try:
            embeddings = []
            for text in texts:
                embedding = await self.aembed_query(text)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            self.logger.error(f"LangChain batch embedding failed: {str(e)}")
            raise


class EmbeddingAdapterFactory:
    """Factory for creating appropriate embedding adapters."""
    
    @staticmethod
    def create_adapter(embedding_model: Any, logger: logging.Logger, 
                      adapter_type: str = "auto") -> EmbeddingAdapter:
        """Create embedding adapter based on model type or specified type."""
        
        if adapter_type == "huggingface":
            return HuggingFaceEmbeddingAdapter(embedding_model, logger)
        elif adapter_type == "openai":
            return OpenAIEmbeddingAdapter(embedding_model, logger)
        elif adapter_type == "langchain":
            return LangChainEmbeddingAdapter(embedding_model, logger)
        else:
            # Auto-detect based on model attributes
            if hasattr(embedding_model, 'model_name') and 'bge' in str(embedding_model.model_name).lower():
                return HuggingFaceEmbeddingAdapter(embedding_model, logger)
            elif hasattr(embedding_model, 'model') and 'openai' in str(embedding_model.model).lower():
                return OpenAIEmbeddingAdapter(embedding_model, logger)
            elif hasattr(embedding_model, 'aembed_query'):
                return LangChainEmbeddingAdapter(embedding_model, logger)
            else:
                # Default to HuggingFace adapter
                return HuggingFaceEmbeddingAdapter(embedding_model, logger) 