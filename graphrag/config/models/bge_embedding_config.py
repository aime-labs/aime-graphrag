# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Configuration for BGE embedding model."""

from pydantic import BaseModel, Field

from graphrag.config.enums import AsyncType, AuthType

from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.enums import ModelType

class BGEEmbeddingConfig(LanguageModelConfig):
    """Configuration for BGE embedding model."""

    type: ModelType = Field(default=ModelType.BGEEmbedding)
    model: str = Field(default="BAAI/bge-m3")
    device: str = Field(default="cpu")
    concurrent_requests: int = Field(default=4)
    async_mode: AsyncType = Field(default=AsyncType.Threaded)
    retry_strategy: str = Field(default="native")
    max_retries: int = Field(default=-1)
    auth_type: AuthType = Field(default=AuthType.None_)
