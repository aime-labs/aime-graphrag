# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Response types for LLM providers."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CompletionUsage:
    """Usage information for a completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletionResponseChoice:
    """A choice in a chat completion response."""

    message: Dict[str, str]
    finish_reason: str
    index: int


@dataclass
class ChatCompletionResponse:
    """Response from a chat completion."""

    choices: List[ChatCompletionResponseChoice]
    model: str
    usage: CompletionUsage


@dataclass
class TextGenerationResponse:
    """Response from text generation."""

    text: str
    model: str
    usage: CompletionUsage


@dataclass
class EmbeddingsResponseItem:
    """An item in an embeddings response."""

    embedding: List[float]
    index: int


@dataclass
class EmbeddingsResponse:
    """Response from embeddings generation."""

    data: List[EmbeddingsResponseItem]
    model: str
