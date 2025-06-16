# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Model Providers module."""

from graphrag.language_model.providers.aime import AimeAPIProvider
from graphrag.language_model.providers.bge import BGEProvider
from graphrag.language_model.providers.fnllm.models import (
    AzureOpenAIChatFNLLM,
    AzureOpenAIEmbeddingFNLLM,
    OpenAIChatFNLLM,
    OpenAIEmbeddingFNLLM,
)

__all__ = [
    "AimeAPIProvider",
    "BGEProvider",
    "OpenAIChatFNLLM",
    "OpenAIEmbeddingFNLLM",
    "AzureOpenAIChatFNLLM",
    "AzureOpenAIEmbeddingFNLLM",
]

PROVIDER_REGISTRY = {
    "OpenAIChatFNLLM": OpenAIChatFNLLM,
    "OpenAIEmbeddingFNLLM": OpenAIEmbeddingFNLLM,
    "AzureOpenAIChatFNLLM": AzureOpenAIChatFNLLM,
    "AzureOpenAIEmbeddingFNLLM": AzureOpenAIEmbeddingFNLLM,
    "AimeAPIProvider": AimeAPIProvider,
    "BGEProvider": BGEProvider,
}
