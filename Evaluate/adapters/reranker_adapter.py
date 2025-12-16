"""
Reranker Adapter for Context Precision Metric

Implements lightweight reranking to reorder GraphRAG artifacts based on semantic
relevance to the question. This ensures the most relevant contexts appear at the
top (Rank 1), satisfying RAGAS assumptions without changing content.

Supports multiple reranker backends:
- BGE-Reranker (BAAI/bge-reranker-large, bge-reranker-base)
- Cross-encoder based rerankers
- Cohere Rerank API
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Tuple, Optional
import asyncio
import logging
import numpy as np


class RerankerAdapter(ABC):
    """Abstract base class for reranker adapters."""
    
    @abstractmethod
    async def arerank(
        self, 
        query: str, 
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: The query text (e.g., the question)
            documents: List of document texts to rerank
            top_k: Optional limit on number of results to return
            
        Returns:
            List of tuples (document, score) sorted by relevance (highest first)
        """
        pass
    
    def rerank_sync(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Synchronous wrapper for arerank."""
        return asyncio.run(self.arerank(query, documents, top_k))


class BGERerankerAdapter(RerankerAdapter):
    """
    Adapter for BAAI BGE Reranker models.
    
    BGE-Reranker is a lightweight cross-encoder that provides high-quality
    relevance scores for passage reranking. It's optimized for both speed
    and accuracy.
    
    Supported models:
    - BAAI/bge-reranker-large (560M params, best quality)
    - BAAI/bge-reranker-base (278M params, balanced)
    - BAAI/bge-reranker-v2-m3 (multilingual)
    """
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "cuda",
        max_length: int = 512,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize BGE Reranker.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda', 'cpu', 'mps')
            max_length: Maximum sequence length for tokenization
            logger: Logger instance
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.logger = logger or logging.getLogger(__name__)
        
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Lazy load the reranker model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            self.logger.info(f"Loading BGE reranker: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            
            # Move to specified device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            elif self.device == "mps" and torch.backends.mps.is_available():
                self.model = self.model.to("mps")
            else:
                self.model = self.model.to("cpu")
                self.device = "cpu"
            
            self.model.eval()
            self.logger.info(f"BGE reranker loaded successfully on {self.device}")
            
        except ImportError as e:
            self.logger.error(
                f"Failed to import required libraries for BGE reranker: {e}. "
                "Please install: pip install transformers torch"
            )
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize BGE reranker: {e}")
            raise
    
    async def arerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents using BGE cross-encoder.
        
        Args:
            query: The query/question text
            documents: List of document/context texts
            top_k: Optional limit on results
            
        Returns:
            Sorted list of (document, score) tuples
        """
        if not documents:
            return []
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._rerank_batch,
            query,
            documents,
            top_k
        )
    
    def _rerank_batch(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Internal method to perform actual reranking."""
        import torch
        
        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]
            
            # Tokenize
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length
                ).to(self.device)
                
                # Get scores
                scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
                scores = scores.cpu().numpy()
            
            # Combine documents with scores
            doc_scores = list(zip(documents, scores))
            
            # Sort by score (descending)
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Apply top_k if specified
            if top_k is not None:
                doc_scores = doc_scores[:top_k]
            
            return doc_scores
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            # Fallback: return original order with neutral scores
            return [(doc, 0.0) for doc in documents]


class CrossEncoderRerankerAdapter(RerankerAdapter):
    """
    Adapter for sentence-transformers cross-encoder models.
    
    Supports models like:
    - cross-encoder/ms-marco-MiniLM-L-6-v2
    - cross-encoder/ms-marco-TinyBERT-L-2-v2
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda",
        logger: Optional[logging.Logger] = None
    ):
        self.model_name = model_name
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            
            self.logger.info(f"Loading cross-encoder: {self.model_name}")
            self.model = CrossEncoder(self.model_name, device=self.device)
            self.logger.info("Cross-encoder loaded successfully")
            
        except ImportError as e:
            self.logger.error(
                f"Failed to import sentence-transformers: {e}. "
                "Please install: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize cross-encoder: {e}")
            raise
    
    async def arerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Rerank using cross-encoder."""
        if not documents:
            return []
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._rerank_batch,
            query,
            documents,
            top_k
        )
    
    def _rerank_batch(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Internal reranking method."""
        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]
            
            # Get scores
            scores = self.model.predict(pairs)
            
            # Combine documents with scores
            doc_scores = list(zip(documents, scores))
            
            # Sort by score (descending)
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Apply top_k if specified
            if top_k is not None:
                doc_scores = doc_scores[:top_k]
            
            return doc_scores
            
        except Exception as e:
            self.logger.error(f"Cross-encoder reranking failed: {e}")
            return [(doc, 0.0) for doc in documents]


class CohereRerankerAdapter(RerankerAdapter):
    """
    Adapter for Cohere Rerank API.
    
    Requires COHERE_API_KEY environment variable.
    """
    
    def __init__(
        self,
        model_name: str = "rerank-english-v3.0",
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        import os
        
        self.model_name = model_name
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.logger = logger or logging.getLogger(__name__)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Cohere client."""
        try:
            import cohere
            
            if not self.api_key:
                raise ValueError("COHERE_API_KEY not found in environment")
            
            self.client = cohere.Client(self.api_key)
            self.logger.info(f"Cohere reranker initialized: {self.model_name}")
            
        except ImportError as e:
            self.logger.error(
                f"Failed to import cohere: {e}. "
                "Please install: pip install cohere"
            )
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Cohere client: {e}")
            raise
    
    async def arerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Rerank using Cohere API."""
        if not documents:
            return []
        
        try:
            response = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model_name,
                top_n=top_k or len(documents)
            )
            
            # Extract results
            results = []
            for result in response.results:
                doc = documents[result.index]
                score = result.relevance_score
                results.append((doc, score))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Cohere reranking failed: {e}")
            return [(doc, 0.0) for doc in documents]


def create_reranker_adapter(
    reranker_type: str,
    model_name: Optional[str] = None,
    device: str = "cuda",
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> RerankerAdapter:
    """
    Factory function to create reranker adapters.
    
    Args:
        reranker_type: Type of reranker ('bge', 'cross-encoder', 'cohere', 'none')
        model_name: Specific model name (optional, uses defaults)
        device: Device for local models
        logger: Logger instance
        **kwargs: Additional arguments for specific adapters
        
    Returns:
        RerankerAdapter instance or None
    """
    logger = logger or logging.getLogger(__name__)
    
    if reranker_type == "none" or reranker_type is None:
        return None
    
    try:
        if reranker_type == "bge":
            model_name = model_name or "BAAI/bge-reranker-large"
            return BGERerankerAdapter(
                model_name=model_name,
                device=device,
                logger=logger,
                **kwargs
            )
        elif reranker_type == "cross-encoder":
            model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            return CrossEncoderRerankerAdapter(
                model_name=model_name,
                device=device,
                logger=logger
            )
        elif reranker_type == "cohere":
            model_name = model_name or "rerank-english-v3.0"
            return CohereRerankerAdapter(
                model_name=model_name,
                logger=logger,
                **kwargs
            )
        else:
            logger.warning(f"Unknown reranker type: {reranker_type}, returning None")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create reranker adapter: {e}")
        return None
