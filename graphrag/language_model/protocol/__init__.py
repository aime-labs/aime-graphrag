# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Base protocol definitions for LLMs."""

from .base import ChatModel, EmbeddingModel
from .response import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    CompletionUsage,
    EmbeddingsResponse,
    EmbeddingsResponseItem,
    TextGenerationResponse,
)

__all__ = [
    "ChatModel",
    "EmbeddingModel",
    "ChatCompletionResponse",
    "ChatCompletionResponseChoice",
    "CompletionUsage",
    "EmbeddingsResponse",
    "EmbeddingsResponseItem",
    "TextGenerationResponse",
]
