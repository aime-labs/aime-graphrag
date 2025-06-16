import logging
import asyncio
from typing import Any, List

try:
    from FlagEmbedding import FlagModel
except ImportError:
    FlagModel = None
    logging.warning("FlagEmbedding is not installed. Please install it with 'pip install FlagEmbedding'.")

class BGEProvider:
    """
    Embedding provider using BGE-M3 via FlagEmbedding.
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu", **kwargs):
        if FlagModel is None:
            raise ImportError("FlagEmbedding is required for BGEProvider.")
        self.model = FlagModel(model_name, query_instruction=None, use_fp16=True, device=device)

    def embed(self, text: str, **kwargs: Any) -> List[float]:
        """
        Generate an embedding for a single text.
        """
        emb = self.model.encode([text])[0]
        return emb.tolist() if hasattr(emb, 'tolist') else list(emb)

    def embed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.
        """
        embs = self.model.encode(text_list)
        return [e.tolist() if hasattr(e, 'tolist') else list(e) for e in embs]

    async def aembed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts asynchronously.
        """
        # Run the synchronous embed_batch in a thread pool
        return await asyncio.to_thread(self.embed_batch, text_list, **kwargs)

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Generate an embedding for a single text asynchronously.
        """
        # Run the synchronous embed in a thread pool
        return await asyncio.to_thread(self.embed, text, **kwargs)
