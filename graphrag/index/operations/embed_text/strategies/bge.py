# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing BGE embedding strategy implementation."""

import asyncio
import logging
from typing import Any, Optional

import numpy as np

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.index.operations.embed_text.strategies.typing import TextEmbeddingResult
from graphrag.index.text_splitting.text_splitting import TokenTextSplitter
from graphrag.index.utils.is_null import is_null
from graphrag.language_model.manager import ModelManager
from graphrag.language_model.protocol.base import EmbeddingModel
from graphrag.logger.progress import ProgressTicker, progress_ticker

log = logging.getLogger(__name__)


async def run(
    input: list[str],
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    args: dict[str, Any],
) -> TextEmbeddingResult:
    """Run the BGE embedding strategy."""
    if is_null(input):
        return TextEmbeddingResult(embeddings=None)

    batch_size = args.get("batch_size", 16)
    llm_config = LanguageModelConfig(**args["llm"])
    
    # Use token-based splitter for BGE
    splitter = TokenTextSplitter(
        chunk_size=512,  # Reasonable default for BGE models
        chunk_overlap=50,
        encoding_name="cl100k_base"  # Default encoding that works with BGE
    )
    
    model = ModelManager().get_or_create_embedding_model(
        name="text_embedding",
        model_type=llm_config.type,
        config=llm_config,
        callbacks=callbacks,
        cache=cache,
    )
    semaphore: asyncio.Semaphore = asyncio.Semaphore(args.get("num_threads", 4))

    # Break up the input texts
    texts, input_sizes = _prepare_embed_texts(input, splitter)
    
    # Process in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = await model.aembed_batch(batch)
        all_embeddings.extend(batch_embeddings)
    
    # Reconstitute the embeddings into the original input texts
    embeddings = _reconstitute_embeddings(all_embeddings, input_sizes)
    return TextEmbeddingResult(embeddings=embeddings)


def _prepare_embed_texts(
    input: list[str], splitter: TokenTextSplitter
) -> tuple[list[str], list[int]]:
    """Split input texts into chunks and return chunks with their sizes."""
    texts: list[str] = []
    sizes: list[int] = []
    
    for text in input:
        if not text or not text.strip():
            sizes.append(0)
            continue
            
        chunks = splitter.split_text(text)
        texts.extend(chunks)
        sizes.append(len(chunks))
    
    return texts, sizes


def _reconstitute_embeddings(
    raw_embeddings: list[list[float]], sizes: list[int]
) -> list[Optional[list[float]]]:
    """Reconstitute the embeddings into the original input texts."""
    embeddings: list[Optional[list[float]]] = []
    start = 0
    
    for size in sizes:
        if size == 0:
            embeddings.append(None)
            continue
            
        chunk_embeddings = raw_embeddings[start : start + size]
        # Use mean pooling for combining chunk embeddings
        mean_embedding = np.mean(chunk_embeddings, axis=0).tolist()
        embeddings.append(mean_embedding)
        start += size
    
    return embeddings
