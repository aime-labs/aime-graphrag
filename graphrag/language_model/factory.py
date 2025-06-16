# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""A package containing a factory for supported llm types."""

from collections.abc import Callable
from typing import Any, ClassVar, Dict, Type, Union

from graphrag.config.enums import ModelType
from graphrag.language_model.protocol import ChatModel, EmbeddingModel
from graphrag.language_model.providers.aime import AimeAPIProvider
from graphrag.language_model.providers.bge import BGEProvider
from graphrag.language_model.providers.fnllm.models import (
    AzureOpenAIChatFNLLM,
    AzureOpenAIEmbeddingFNLLM,
    OpenAIChatFNLLM,
    OpenAIEmbeddingFNLLM,
)


class ModelFactory:
    """Factory for creating language model instances."""

    _chat_models: Dict[str, Type[ChatModel]] = {
        "aime_chat": AimeAPIProvider,
        "openai_chat_fnllm": OpenAIChatFNLLM,
        "azure_openai_chat_fnllm": AzureOpenAIChatFNLLM,
    }

    _embedding_models: Dict[str, Type[EmbeddingModel]] = {
        "openai_embedding_fnllm": OpenAIEmbeddingFNLLM,
        "azure_openai_embedding_fnllm": AzureOpenAIEmbeddingFNLLM,
        "bge_embedding": BGEProvider,
    }

    @classmethod
    def register_chat(cls, model_type: ModelType, model_class: Type[ChatModel]) -> None:
        """Register a chat model type.

        Args:
            model_type: The type of model to register.
            model_class: The model class to register.
        """
        cls._chat_models[model_type.value] = model_class

    @classmethod
    def register_embedding(
        cls, model_type: ModelType, model_class: Type[EmbeddingModel]
    ) -> None:
        """Register an embedding model type.

        Args:
            model_type: The type of model to register.
            model_class: The model class to register.
        """
        cls._embedding_models[model_type.value] = model_class

    @classmethod
    def create_chat_model(cls, model_type: str, **kwargs: Any) -> ChatModel:
        """Create a chat model instance.

        Args:
            model_type: The type of model to create.
            **kwargs: Additional parameters for instantiation.

        Returns:
            A chat model instance.

        Raises:
            ValueError: If the model type is not supported.
        """
        if model_type not in cls._chat_models:
            raise ValueError(
                f"Unsupported chat model type: {model_type}. "
                f"Supported types: {list(cls._chat_models.keys())}"
            )
        return cls._chat_models[model_type](**kwargs)

    @classmethod
    def create_embedding_model(cls, model_type: str, **kwargs: Any) -> EmbeddingModel:
        """Create an embedding model instance.

        Args:
            model_type: The type of model to create.
            **kwargs: Additional parameters for instantiation.

        Returns:
            An embedding model instance.

        Raises:
            ValueError: If the model type is not supported.
        """
        if model_type == "bge_embedding":
            from graphrag.config.models.bge_embedding_config import BGEEmbeddingConfig
            config = BGEEmbeddingConfig(**kwargs)
            return cls._embedding_models[model_type](
                model_name=config.model,
                device=config.device,
                concurrent_requests=config.concurrent_requests,
                async_mode=config.async_mode,
                retry_strategy=config.retry_strategy,
                max_retries=config.max_retries
            )
        
        if model_type not in cls._embedding_models:
            raise ValueError(
                f"Unsupported embedding model type: {model_type}. "
                f"Supported types: {list(cls._embedding_models.keys())}"
            )
        return cls._embedding_models[model_type](**kwargs)

    @classmethod
    def is_supported_chat_model(cls, model_type: str) -> bool:
        """Check if a chat model type is supported.

        Args:
            model_type: The type of model to check.

        Returns:
            True if the model type is supported, False otherwise.
        """
        return model_type in cls._chat_models

    @classmethod
    def is_supported_embedding_model(cls, model_type: str) -> bool:
        """Check if an embedding model type is supported.

        Args:
            model_type: The type of model to check.

        Returns:
            True if the model type is supported, False otherwise.
        """
        return model_type in cls._embedding_models

    @classmethod
    def get_chat_models(cls) -> list[str]:
        """Get a list of supported chat model types.

        Returns:
            A list of supported chat model types.
        """
        return list(cls._chat_models.keys())

    @classmethod
    def get_embedding_models(cls) -> list[str]:
        """Get a list of supported embedding model types.

        Returns:
            A list of supported embedding model types.
        """
        return list(cls._embedding_models.keys())

    @classmethod
    def is_supported_model(cls, model_type: Union[str, ModelType]) -> bool:
        """Check if the given model type is supported.

        Args:
            model_type: The model type to check, either as a string or ModelType enum.

        Returns:
            True if the model type is supported, False otherwise.
        """
        model_type_str = (
            model_type.value if isinstance(model_type, ModelType) else model_type
        )
        return cls.is_supported_chat_model(
            model_type_str
        ) or cls.is_supported_embedding_model(model_type_str)


# --- Register default implementations ---
ModelFactory.register_chat(ModelType.AIMEChat, AimeAPIProvider)
ModelFactory.register_embedding(ModelType.OpenAIEmbedding, OpenAIEmbeddingFNLLM)
ModelFactory.register_embedding(ModelType.AzureOpenAIEmbedding, AzureOpenAIEmbeddingFNLLM)

# --- Register BGE embedding ---
ModelFactory.register_embedding(ModelType.BGEEmbedding, BGEProvider)
