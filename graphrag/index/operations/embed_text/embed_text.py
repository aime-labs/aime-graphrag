# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing embed_text, load_strategy and create_row_from_embedding_data methods definition."""

import logging
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.embeddings import create_collection_name
from graphrag.index.operations.embed_text.strategies.typing import TextEmbeddingStrategy
from graphrag.vector_stores.base import BaseVectorStore, VectorStoreDocument
from graphrag.vector_stores.factory import VectorStoreFactory
from graphrag.index.operations.embed_text.strategies import TextEmbedStrategyType, load_strategy

log = logging.getLogger(__name__)

# Per Azure OpenAI Limits
# https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
DEFAULT_EMBEDDING_BATCH_SIZE = 500


# Keep the old enum for backward compatibility
class LegacyTextEmbedStrategyType(str, Enum):
    """Legacy TextEmbedStrategyType class definition."""
    openai = "openai"
    mock = "mock"
    bge = "bge"

    def __repr__(self) -> str:
        """Get a string representation."""
        return f"{self.__class__.__name__}.{self.name}"


class TextEmbedStrategyType(str, Enum):
    """TextEmbedStrategyType class definition."""
    OPENAI = "openai"
    BGE = "bge"
    MOCK = "mock"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


async def embed_text(
    input: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    embed_column: str,
    strategy: dict,
    embedding_name: str,
    id_column: str = "id",
    title_column: str | None = None,
):
    """Embed a piece of text into a vector space. The operation outputs a new column containing a mapping between doc_id and vector."""
    vector_store_config = strategy.get("vector_store")

    if vector_store_config:
        collection_name = _get_collection_name(vector_store_config, embedding_name)
        vector_store: BaseVectorStore = _create_vector_store(
            vector_store_config, collection_name
        )
        vector_store_workflow_config = vector_store_config.get(
            embedding_name, vector_store_config
        )
        return await _text_embed_with_vector_store(
            input=input,
            callbacks=callbacks,
            cache=cache,
            embed_column=embed_column,
            strategy=strategy,
            vector_store=vector_store,
            vector_store_config=vector_store_workflow_config,
            id_column=id_column,
            title_column=title_column,
        )

    return await _text_embed_in_memory(
        input=input,
        callbacks=callbacks,
        cache=cache,
        embed_column=embed_column,
        strategy=strategy,
    )


def _get_strategy_executor(strategy: dict):
    """Get the appropriate strategy executor based on the strategy config."""
    strategy_type = strategy.get("type", "")
    
    # Check if this is a BGE embedding strategy
    if (strategy_type == "bge" or 
        (isinstance(strategy.get("llm"), dict) and 
         strategy["llm"].get("type", "").lower() == "bge_embedding")):
        return load_strategy(TextEmbedStrategyType.BGE)
    
    # Default to OPENAI for backward compatibility
    return load_strategy(TextEmbedStrategyType.OPENAI)


async def _text_embed_in_memory(
    input: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    embed_column: str,
    strategy: dict,
):
    strategy_exec = _get_strategy_executor(strategy)
    strategy_config = {**strategy}

    texts: list[str] = input[embed_column].to_numpy().tolist()
    result = await strategy_exec(texts, callbacks, cache, strategy_config)

    return result.embeddings if result else None


async def _text_embed_with_vector_store(
    input: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    embed_column: str,
    strategy: dict[str, Any],
    vector_store: BaseVectorStore,
    vector_store_config: dict,
    id_column: str = "id",
    title_column: str | None = None,
):
    strategy_exec = _get_strategy_executor(strategy)
    strategy_config = {**strategy}

    # if max_retries is not set, inject a dynamically assigned value based on the total number of expected LLM calls to be made
    if strategy_config.get("llm") and strategy_config["llm"]["max_retries"] == -1:
        strategy_config["llm"]["max_retries"] = len(input)

    # Get vector-storage configuration
    insert_batch_size: int = (
        vector_store_config.get("batch_size") or DEFAULT_EMBEDDING_BATCH_SIZE
    )

    overwrite: bool = vector_store_config.get("overwrite", True)

    if embed_column not in input.columns:
        msg = f"Column {embed_column} not found in input dataframe with columns {input.columns}"
        raise ValueError(msg)
    title = title_column or embed_column
    if title not in input.columns:
        msg = (
            f"Column {title} not found in input dataframe with columns {input.columns}"
        )
        raise ValueError(msg)
    if id_column not in input.columns:
        msg = f"Column {id_column} not found in input dataframe with columns {input.columns}"
        raise ValueError(msg)

    total_rows = 0
    for row in input[embed_column]:
        if isinstance(row, list):
            total_rows += len(row)
        else:
            total_rows += 1

    i = 0
    starting_index = 0

    all_results = []

    while insert_batch_size * i < input.shape[0]:
        batch = input.iloc[insert_batch_size * i : insert_batch_size * (i + 1)]
        texts: list[str] = batch[embed_column].to_numpy().tolist()
        titles: list[str] = batch[title].to_numpy().tolist()
        ids: list[str] = batch[id_column].to_numpy().tolist()
        result = await strategy_exec(texts, callbacks, cache, strategy_config)
        if result.embeddings:
            embeddings = [
                embedding for embedding in result.embeddings if embedding is not None
            ]
            all_results.extend(embeddings)

        vectors = result.embeddings or []
        documents: list[VectorStoreDocument] = []
        for doc_id, doc_text, doc_title, doc_vector in zip(
            ids, texts, titles, vectors, strict=True
        ):
            if type(doc_vector) is np.ndarray:
                doc_vector = doc_vector.tolist()
            document = VectorStoreDocument(
                id=doc_id,
                text=doc_text,
                vector=doc_vector,
                attributes={"title": doc_title},
            )
            documents.append(document)

        vector_store.load_documents(documents, overwrite and i == 0)
        starting_index += len(documents)
        i += 1

    return all_results


def _create_vector_store(
    vector_store_config: dict, collection_name: str
) -> BaseVectorStore:
    vector_store_type: str = str(vector_store_config.get("type"))
    if collection_name:
        vector_store_config.update({"collection_name": collection_name})

    vector_store = VectorStoreFactory().create_vector_store(
        vector_store_type, kwargs=vector_store_config
    )

    vector_store.connect(**vector_store_config)
    return vector_store


def _get_collection_name(vector_store_config: dict, embedding_name: str) -> str:
    container_name = vector_store_config.get("container_name", "default")
    collection_name = create_collection_name(container_name, embedding_name)

    msg = f"using vector store {vector_store_config.get('type')} with container_name {container_name} for embedding {embedding_name}: {collection_name}"
    log.info(msg)
    return collection_name


def load_strategy(strategy: TextEmbedStrategyType) -> TextEmbeddingStrategy:
    """Load strategy method definition."""
    match strategy:
        case TextEmbedStrategyType.OPENAI:
            from graphrag.index.operations.embed_text.strategies.openai import (
                run as run_openai,
            )
            return run_openai
        case TextEmbedStrategyType.BGE:
            from graphrag.index.operations.embed_text.strategies.bge import (
                run as run_bge,
            )
            return run_bge
        case TextEmbedStrategyType.MOCK:
            from graphrag.index.operations.embed_text.strategies.mock import (
                run as run_mock,
            )
            return run_mock
        case _:
            # For backward compatibility with old enum values
            if str(strategy).lower() == "bge":
                from graphrag.index.operations.embed_text.strategies.bge import (
                    run as run_bge,
                )
                return run_bge
            # Default to OPENAI for backward compatibility
            from graphrag.index.operations.embed_text.strategies.openai import (
                run as run_openai,
            )
            return run_openai
