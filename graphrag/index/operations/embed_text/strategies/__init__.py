# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Indexing Engine embed strategies package root."""

from enum import Enum
from typing import Any, Callable, Optional

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.index.operations.embed_text.strategies.typing import TextEmbeddingResult


class TextEmbedStrategyType(str, Enum):
    """Text embedding strategy types."""
    OPENAI = "openai"
    BGE = "bge"
    MOCK = "mock"


def load_strategy(strategy: TextEmbedStrategyType) -> Callable[[list[str], WorkflowCallbacks, PipelineCache, dict[str, Any]], Any]:
    """Load the specified embedding strategy."""
    if strategy == TextEmbedStrategyType.OPENAI:
        from .openai import run as openai_run
        return openai_run
    elif strategy == TextEmbedStrategyType.BGE:
        from .bge import run as bge_run
        return bge_run
    elif strategy == TextEmbedStrategyType.MOCK:
        from .mock import run as mock_run
        return mock_run
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
