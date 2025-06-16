# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""AIME API provider implementation."""

import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import aiohttp
from aime_api_client_interface import ModelAPI

from graphrag.language_model.protocol import (
    CompletionUsage,
    TextGenerationResponse,
)
from graphrag.language_model.protocol.base import ChatModel, EmbeddingModel
from graphrag.language_model.response.base import (
    BaseModelOutput,
    BaseModelResponse,
    ModelResponse,
)

log = logging.getLogger(__name__)


# Extend TextGenerationResponse to include required attributes
@dataclass
class EnhancedTextGenerationResponse(TextGenerationResponse):
    """Enhanced TextGenerationResponse with additional attributes for GraphRAG compatibility."""

    output: BaseModelOutput = field(default_factory=lambda: BaseModelOutput(content=""))
    history: List[Dict[str, str]] = field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = field(default=None)
    parsed_response: Optional[Any] = field(default=None)

    def __post_init__(self):
        """Initialize additional fields after dataclass initialization."""
        self.output = BaseModelOutput(content=self.text)
        if not hasattr(self, "history"):
            self.history = []
        if not hasattr(self, "metrics"):
            self.metrics = {}


class AimeAPIProvider(ChatModel, EmbeddingModel):
    """AIME API provider implementation."""

    def __init__(
        self,
        api_url: str | None = None,
        model_name: str | None = None,
        email: str | None = None,
        api_key: str | None = None,
        config: Any | None = None,
        **kwargs: Any,
    ):
        """Initialize the AIME API provider.

        Args:
            api_url: The AIME API URL.
            model_name: The model name to use.
            email: The email to use for authentication.
            api_key: The API key to use for authentication.
            config: The language model configuration.
            **kwargs: Additional keyword arguments.
        """
        if config is not None:
            self.api_url = config.api_base or api_url or "https://api.aime.info/"
            self.model_name = config.model or model_name
            self.email = config.api_key or email
            self.api_key = config.api_key or api_key
        else:
            self.api_url = api_url or "https://api.aime.info/"
            self.model_name = model_name
            self.email = email
            self.api_key = api_key

        if not self.email or not self.api_key:
            raise ValueError("Email and API key are required for AIME API")

        self.model_api = ModelAPI(
            api_server=self.api_url,
            endpoint_name=self.model_name,
            user=self.email,
            key=self.api_key,
        )
        self._session_token = None
        self._semaphore = asyncio.Semaphore(kwargs.get("max_concurrent_requests", 10))

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        if hasattr(self.model_api, "session") and self.model_api.session:
            await self.model_api.close_session()
        self._session_token = None

    def __del__(self):
        """Ensure cleanup on deletion."""
        if (
            hasattr(self.model_api, "session")
            and self.model_api.session
            and not self.model_api.session.closed
        ):
            asyncio.run(self.model_api.close_session())

    async def call_with_retries_async(
        self, func, *args, max_retries=5, base_delay=1.0, **kwargs
    ):
        """Call a function with retries and exponential backoff."""
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff = base_delay * (2**attempt) + random.uniform(0, 0.1)
                    log.warning(f"Retrying after {backoff:.2f}s due to error: {e}")
                    await asyncio.sleep(backoff)
                else:
                    log.error(f"Max retries reached. Error: {e}")
                    raise

    def call_with_retries_sync(
        self, func, *args, max_retries=5, base_delay=1.0, **kwargs
    ):
        return func(*args, **kwargs)

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> ModelResponse:
        """Generate a response for the given text.

        Args:
            prompt: The text to generate a response for.
            history: The conversation history.
            **kwargs: Additional keyword arguments (e.g., model parameters).

        Returns:
            A TextGenerationResponse containing the response.
        """
        if not self._session_token:
            if not hasattr(self.model_api, "session") or self.model_api.session is None:
                self.model_api.session = aiohttp.ClientSession()
            self._session_token = await self.model_api.do_api_login_async()
            self.model_api.client_session_auth_key = self._session_token

        if history is None:
            chat_context = [
                {
                    "role": "system",
                    "content": "You are a helpful, respectful and honest assistant.",
                }
            ]
        else:
            chat_context = [
                msg if isinstance(msg, dict) else {"role": "user", "content": str(msg)}
                for msg in history
            ]

        chat_context_str = json.dumps(chat_context)

        base_params = {
            "prompt_input": prompt,
            "chat_context": chat_context_str,
            "top_k": kwargs.get("top_k", 40),
            "top_p": kwargs.get("top_p", 0.9),
            "temperature": kwargs.get("temperature", 0.8),
            "max_gen_tokens": kwargs.get("max_gen_tokens", 1000),
            "wait_for_result": kwargs.get("wait_for_result", True),
            "client_session_auth_key": self._session_token,
        }

        valid_extra_params = {
            "format",
            "stream",
            "timeout",
            "max_retries",
            "retry_interval",
            "retry_on_http_status",
            "retry_on_socketio_status",
            "endpoint",
        }

        api_params = {
            k: v for k, v in kwargs.items() if k in valid_extra_params and v is not None
        }
        base_params.update(api_params)

        try:
            text = ""
            usage = {}
            metrics = {}

            async with self._semaphore:
                output_generator = self.model_api.get_api_request_generator(base_params)
                async for progress in output_generator:
                    if not isinstance(progress, dict):
                        continue

                    if not progress.get("success", False):
                        error_msg = (
                            progress.get("error")
                            or progress.get("description")
                            or "Unknown error"
                        )
                        raise ValueError(f"AIME API request failed: {error_msg}")

                    job_state = progress.get("job_state")

                    if job_state == "done":
                        result_data = progress.get("result_data", {})
                        if isinstance(result_data, dict):
                            text = result_data.get("text", "")
                            metrics = {
                                "num_generated_tokens": result_data.get(
                                    "num_generated_tokens"
                                ),
                                "compute_duration": result_data.get("compute_duration"),
                                "total_duration": result_data.get("total_duration"),
                                "prompt_length": result_data.get("prompt_length"),
                                "max_seq_len": result_data.get("max_seq_len"),
                                "model_name": result_data.get("model_name"),
                                "worker_interface_version": result_data.get(
                                    "worker_interface_version"
                                ),
                            }
                            usage = {
                                "prompt_tokens": result_data.get("prompt_length", 0),
                                "completion_tokens": result_data.get(
                                    "num_generated_tokens", 0
                                ),
                                "total_tokens": result_data.get("prompt_length", 0)
                                + result_data.get("num_generated_tokens", 0),
                            }
                            break

            if not text:
                raise ValueError("No response text received from AIME API")

            usage_obj = CompletionUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

            return EnhancedTextGenerationResponse(
                text=text,
                model=self.model_name,
                usage=usage_obj,
                output=BaseModelOutput(content=text),
                history=history or [],
                metrics=metrics,
            )
        except Exception as e:
            log.error(f"AIME API request failed: {str(e)}")
            raise

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Generate a response for the given text using a streaming interface.

        Args:
            prompt: The text to generate a response for.
            history: The conversation history.
            **kwargs: Additional keyword arguments (e.g., model parameters).

        Yields:
            Strings representing the response.
        """
        if not self._session_token:
            if not hasattr(self.model_api, "session") or self.model_api.session is None:
                self.model_api.session = aiohttp.ClientSession()
            self._session_token = await self.model_api.do_api_login_async()
            self.model_api.client_session_auth_key = self._session_token
        chat_context = (
            [
                {
                    "role": "system",
                    "content": "You are a helpful, respectful and honest assistant.",
                }
            ]
            if history is None
            else history
        )
        chat_context.append({"role": "user", "content": prompt})
        chat_context_str = json.dumps(chat_context)
        
        params = {
            "prompt_input": prompt,
            "chat_context": chat_context_str,
            "top_k": int(kwargs.get("top_k", 40)),
            "top_p": float(kwargs.get("top_p", 0.9)),
            "temperature": float(kwargs.get("temperature", 0.8)),
            "max_gen_tokens": int(kwargs.get("max_gen_tokens", 1000)),
            "wait_for_result": False,
            "client_session_auth_key": self._session_token,
        }
        
        for k, v in kwargs.items():
            if k not in params and v is not None and k not in ['stream', 'model_parameters']:
                params[k] = v
        
        try:
            generator = self.model_api.get_api_request_generator(
                params=params,
                progress_interval=0.1,
                output_format='text'  # Use text format for streaming
            )
            
            buffer = ""
            response_received = False
            
            async for chunk in generator:
                if not chunk:
                    continue
                    
                response_received = True
                
                # Ensure chunk is a dictionary
                if not isinstance(chunk, dict):
                    log.warning(f"Received non-dictionary chunk: {chunk}")
                    continue
                
                # Check for error responses
                if not chunk.get("success", True):
                    error_msg = (
                        chunk.get("error") or 
                        chunk.get("description") or 
                        str(chunk.get("errors", ["Unknown error"])) or
                        f"Invalid response format: {chunk}"
                    )
                    log.error(f"AIME API error: {error_msg}")
                    raise ValueError(f"AIME API stream failed: {error_msg}")
                
                job_state = chunk.get("job_state")
                
                # Process progress data for streaming
                if job_state == "processing":
                    progress_data = chunk.get("progress_data", {})
                    if isinstance(progress_data, dict) and "text" in progress_data:
                        new_text = progress_data["text"]
                        # Only yield new content that hasn't been sent yet
                        if new_text.startswith(buffer):
                            new_content = new_text[len(buffer):]
                            if new_content:
                                yield new_content
                                buffer = new_text
                # Process final result
                elif job_state == "done":
                    result_data = chunk.get("result_data", {})
                    if isinstance(result_data, dict) and "text" in result_data:
                        final_text = result_data["text"]
                        # Yield any remaining content that hasn't been sent
                        if final_text.startswith(buffer):
                            remaining_content = final_text[len(buffer):]
                            if remaining_content:
                                yield remaining_content
                        break
            
            if not response_received:
                error_msg = "No valid response received from the API"
                log.error(error_msg)
                raise ValueError(error_msg)
                    
        except json.JSONDecodeError as je:
            error_msg = f"Failed to decode JSON response: {str(je)}"
            log.error(f"{error_msg}. Response: {chunk if 'chunk' in locals() else 'No chunk data'}")
            raise ValueError(error_msg) from je
        except Exception as e:
            error_msg = f"Unexpected error in achat_stream: {str(e)}"
            log.error(error_msg)
            log.error(f"Exception type: {type(e).__name__}")
            try:
                stack_trace = traceback.format_exc()
                log.error(f"Stack trace: {stack_trace}")
            except Exception as traceback_error:
                log.error(f"Failed to get stack trace: {traceback_error}")
            
            # For BrokenPipeError, provide a more helpful message
            if isinstance(e, BrokenPipeError):
                error_msg = "Connection to AIME API was lost. This might be due to network issues or server problems. Please try again later."
                log.error(error_msg)
            
            raise ValueError(error_msg) from e

    def chat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> TextGenerationResponse:
        """Generate a response for the given text.

        Args:
            prompt: The text to generate a response for.
            history: The conversation history.
            **kwargs: Additional keyword arguments (e.g., model parameters).

        Returns:
            A TextGenerationResponse containing the response.
        """
        return self.call_with_retries_sync(
            lambda *a, **kw: asyncio.run(self.achat(*a, **kw)),
            prompt,
            history,
            **kwargs,
        )

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Generate a response for the given text using a streaming interface.

        Args:
            prompt: The text to generate a response for.
            history: The conversation history.
            **kwargs: Additional keyword arguments (e.g., model parameters).

        Yields:
            Strings representing the response.
        """
        return self.achat_stream(prompt, history, **kwargs)

    async def aembed_batch(
        self, text_list: list[str], **kwargs: Any
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            text_list: The list of texts to generate embeddings for.
            **kwargs: Additional keyword arguments (e.g., model parameters).

        Returns:
            A list of embedding vectors.
        """
        if not self._session_token:
            if not hasattr(self.model_api, "session") or self.model_api.session is None:
                self.model_api.session = aiohttp.ClientSession()
            self._session_token = await self.model_api.do_api_login_async()
            self.model_api.client_session_auth_key = self._session_token
        params = {
            "input": text_list,
            "model": kwargs.get("model", "BAAI/bge-small-en-v1.5"),
            "wait_for_result": kwargs.get("wait_for_result", True),
            "client_session_auth_key": self._session_token,
        }
        params.update({k: v for k, v in kwargs.items() if k not in params})
        result = await self.call_with_retries_async(
            self.model_api.do_api_request_async, params, endpoint="embeddings"
        )
        if result and result.get("success") and "embeddings" in result:
            return result["embeddings"]
        else:
            return [[]] * len(text_list)

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to generate an embedding for.
            **kwargs: Additional keyword arguments (e.g., model parameters).

        Returns:
            An embedding vector.
        """
        embeddings = await self.aembed_batch([text], **kwargs)
        return embeddings[0] if embeddings else []

    def embed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            text_list: The list of texts to generate embeddings for.
            **kwargs: Additional keyword arguments (e.g., model parameters).

        Returns:
            A list of embedding vectors.
        """
        return self.call_with_retries_sync(
            lambda *a, **kw: asyncio.run(self.aembed_batch(*a, **kw)),
            text_list,
            **kwargs,
        )

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to generate an embedding for.
            **kwargs: Additional keyword arguments (e.g., model parameters).

        Returns:
            An embedding vector.
        """
        return self.call_with_retries_sync(
            lambda *a, **kw: asyncio.run(self.aembed(*a, **kw)), text, **kwargs
        )
