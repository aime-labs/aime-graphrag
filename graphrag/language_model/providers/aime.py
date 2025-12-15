# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""AIME API provider implementation."""

import asyncio
import inspect
import json
import logging
import time
import traceback
import random
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import aiohttp
from aiohttp.client_exceptions import ServerDisconnectedError
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
            self.email = getattr(config, 'email', None) or email
            self.api_key = config.api_key or api_key
            self.chat_output_format = getattr(config, 'chat_output_format', None)
        else:
            self.api_url = api_url or "https://api.aime.info/"
            self.model_name = model_name
            self.email = email
            self.api_key = api_key
            self.chat_output_format = kwargs.get('chat_output_format', None)

        if not self.email or not self.api_key:
            raise ValueError("Email and API key are required for AIME API")

        # Normalize api_url to avoid double slashes
        api_url_clean = self.api_url.rstrip('/')
        
        # Create ModelAPI with compatibility for different client versions
        try:
            sig = inspect.signature(ModelAPI.__init__)
            params = sig.parameters
            
            if 'api_key' in params:
                self.model_api = ModelAPI(
                    api_server=api_url_clean,
                    endpoint_name=self.model_name,
                    user=self.email,
                    api_key=self.api_key,
                )
            elif 'key' in params:
                self.model_api = ModelAPI(
                    api_server=api_url_clean,
                    endpoint_name=self.model_name,
                    user=self.email,
                    api_key=self.api_key,
                )
            else:
                self.model_api = ModelAPI(
                    api_url_clean,
                    self.model_name,
                    self.email,
                    self.api_key,
                )
        except Exception as e:
            log.error(f"Failed to instantiate ModelAPI: {e}")
            raise
            
        self._session_token = None
        # Get concurrent_requests from config or kwargs
        concurrent = getattr(config, "concurrent_requests", None) or kwargs.get("concurrent_requests", 15)
        self._semaphore = asyncio.Semaphore(concurrent)

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
            hasattr(self, "model_api")
            and hasattr(self.model_api, "session")
            and self.model_api.session
            and not self.model_api.session.closed
        ):
            try:
                # Try to get the current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule the cleanup instead
                    loop.create_task(self.model_api.close_session())
                else:
                    # If no loop is running, safe to use asyncio.run()
                    asyncio.run(self.model_api.close_session())
            except RuntimeError:
                # No event loop available, skip cleanup
                pass

    async def call_with_retries_async(
        self, func, *args, max_retries=5, base_delay=3.0, **kwargs
    ):
        """Call a function with retries and exponential backoff."""
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError, ServerDisconnectedError, ConnectionResetError, BrokenPipeError) as e:
                if attempt < max_retries - 1:
                    backoff = base_delay * (2**attempt) + random.uniform(0, 0.1)
                    log.warning(f"Connection error on attempt {attempt + 1}, retrying after {backoff:.2f}s: {e}")
                    await asyncio.sleep(backoff)
                    # Reset session for connection errors
                    if hasattr(self.model_api, "session") and self.model_api.session:
                        await self.model_api.close_session()
                        self.model_api.session = None
                        self._session_token = None
                else:
                    log.error(f"Max retries reached for connection error: {e}")
                    raise
            except Exception as e:
                error_str = str(e).lower()
                # Check if this is a retryable error
                is_retryable = (
                    "timeout" in error_str or
                    "network" in error_str or
                    "session" in error_str or
                    "connection" in error_str or
                    "no free queue slot" in error_str or
                    "queue slot" in error_str or
                    "status code: 400" in error_str or
                    "status code: 429" in error_str or
                    "status code: 503" in error_str
                )
                
                if is_retryable and attempt < max_retries - 1:
                    backoff = base_delay * (2**attempt) + random.uniform(0, 0.1)
                    log.warning(f"Retryable error on attempt {attempt + 1}/{max_retries}, retrying after {backoff:.2f}s: {e}")
                    await asyncio.sleep(backoff)
                    # Reset session for queue slot errors to force re-login
                    if "queue slot" in error_str and hasattr(self.model_api, "session"):
                        if self.model_api.session:
                            await self.model_api.close_session()
                        self.model_api.session = None
                        self._session_token = None
                elif not is_retryable:
                    # Non-retryable error, fail immediately
                    log.error(f"Non-retryable error: {e}")
                    raise
                else:
                    # Max retries reached for retryable error
                    log.error(f"Max retries ({max_retries}) reached. Error: {e}")
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
            self._session_token = await self.model_api.do_api_login_async(
                user=self.email,
                api_key=self.api_key,
            )
            self.model_api.client_session_auth_key = self._session_token

        # Create chat context with recommended AIME API format
        assistant_name = "GraphRAG Assistant"
        
        if history is None:
            chat_context = [
                {
                    "role": "system",
                    "content": (
                        f"You are a helpful, respectful and honest assistant named {assistant_name}. "
                        "You specialize in knowledge analysis, data interpretation, and providing insights from GraphRAG (Graph Retrieval-Augmented Generation) systems. "
                        "Always answer as helpfully as possible, while being safe. "
                        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                        "Please ensure that your responses are socially unbiased and positive in nature. "
                        "When analyzing data or providing insights, be precise and cite your sources when available. "
                        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                        "If you don't know the answer to a question, please don't share false information. "
                        "Focus on providing accurate, data-driven insights based on the information available to you."
                    )
                },
                {
                    "role": "user", 
                    "content": f"Hello, {assistant_name}."
                },
                {
                    "role": "assistant", 
                    "content": "Hello! I'm your GraphRAG Assistant, specialized in knowledge analysis and data interpretation. How can I help you analyze data or extract insights today?"
                }
            ]
        else:
            # For existing history, preserve the system prompt structure but ensure proper format
            chat_context = []
            system_prompt_added = False
            
            for msg in history:
                if isinstance(msg, dict):
                    chat_context.append(msg)
                    if msg.get("role") == "system":
                        system_prompt_added = True
                else:
                    chat_context.append({"role": "user", "content": str(msg)})
            
            # Add system prompt if not already present
            if not system_prompt_added:
                system_msg = {
                    "role": "system",
                    "content": (
                        f"You are a helpful, respectful and honest assistant named {assistant_name}. "
                        "You specialize in knowledge analysis, data interpretation, and providing insights from GraphRAG systems. "
                        "Always answer as helpfully as possible, while being safe and accurate."
                    )
                }
                chat_context.insert(0, system_msg)

        # Add user's question/task to the chat context according to AIME API recommendations
        # Handle json parameter - if json=True, modify the prompt to request JSON format
        modified_prompt = prompt
        if kwargs.get("json", False):
            # Add JSON instruction to the prompt if it's not already there
            if "JSON" not in prompt.upper() and "json" not in prompt.lower():
                json_instruction = "\n\nIMPORTANT: Please respond ONLY with valid JSON format. Do not include any explanatory text before or after the JSON. Start your response with { and end with }."
                modified_prompt = prompt + json_instruction

        # Add the user prompt to chat_context as recommended by AIME API
        chat_context.append({"role": "user", "content": modified_prompt})
        
        chat_context_str = json.dumps(chat_context)

        # Extract model_parameters if provided (nested dict from GraphRAG)
        model_params = kwargs.get("model_parameters", {}) or {}
        
        base_params = {
            "chat_context": chat_context_str,  # Use chat_context instead of prompt_input for better compatibility
            "top_k": int(model_params.get("top_k", kwargs.get("top_k", 40))),
            "top_p": float(model_params.get("top_p", kwargs.get("top_p", 0.9))),
            "temperature": float(model_params.get("temperature", kwargs.get("temperature", 0.1))),
            "max_gen_tokens": int(model_params.get("max_gen_tokens", kwargs.get("max_gen_tokens", kwargs.get("max_tokens", 3500)))),
            "wait_for_result": kwargs.get("wait_for_result", True),
            "client_session_auth_key": self._session_token,
        }
        
        # Add chat_output_format if specified (to avoid <think> tags in responses)
        if self.chat_output_format:
            base_params["chat_output_format"] = self.chat_output_format
            log.debug(f"AIME achat - Setting chat_output_format to: {self.chat_output_format}")
        
        # Log the parameters being used for debugging
        log.debug(f"AIME achat - model_params from caller: {model_params}")
        log.debug(f"AIME achat - Final params: temperature={base_params['temperature']}, max_gen_tokens={base_params['max_gen_tokens']}, top_p={base_params['top_p']}")

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
            # Use retry mechanism for the entire API call
            result = await self.call_with_retries_async(self._make_api_request, base_params)
            return result
        except Exception as e:
            log.error(f"AIME API request failed after retries: {str(e)}")
            raise

    async def _make_api_request(self, base_params: dict) -> EnhancedTextGenerationResponse:
        """Make the actual API request with proper error handling."""
        text = ""
        usage = {}
        metrics = {}

        async with self._semaphore:
            output_generator = self.model_api.get_api_request_generator(base_params)
            
            # Add 2-minute timeout to prevent indefinite hangs
            async def process_api_response():
                """Process API response with timeout protection."""
                nonlocal text, usage, metrics
                async for progress in output_generator:
                    if not isinstance(progress, dict):
                        continue

                    if not progress.get("success", False):
                        error_msg = (
                            progress.get("error")
                            or progress.get("description")
                            or "Unknown error"
                        )
                        log.warning(f"AIME API request failed: {error_msg}")
                        # Don't immediately raise error, continue to see if we get any partial response
                        continue

                    job_state = progress.get("job_state")

                    if job_state == "done":
                        result_data = progress.get("result_data", {})
                        log.debug(f"AIME API result_data: {result_data}")
                        if isinstance(result_data, dict):
                            # IMPORTANT: First check if there's an error in the response
                            # The API may return 'error' key when request fails (e.g., context too large)
                            if result_data.get("error"):
                                error_msg = result_data.get("error")
                                log.error(f"AIME API returned error in result_data: {error_msg}")
                                # Don't continue processing, we know this request failed
                                # Fall through to the warning handler with empty text
                                text = ""
                            else:
                                text = result_data.get("text", "")
                                # Check if text is empty and try alternative fields
                                if not text:
                                    # Try alternative field names that might contain the response
                                    text = (
                                        result_data.get("output", "") or
                                        result_data.get("response", "") or
                                        result_data.get("content", "") or
                                        str(result_data.get("result", ""))
                                    )
                                    log.debug(f"Text was empty, tried alternatives. Found: {text[:40] if text else 'None'}")
                                
                            # If still no text, try to extract from nested structures
                            if not text and result_data:
                                # Check if result_data itself contains the text
                                if "text" in str(result_data).lower():
                                    # Try to extract text from string representation
                                    # IMPORTANT: Exclude known metadata keys that should NOT be used as text
                                    # These are API response metadata fields, not actual content
                                    metadata_keys = {
                                        'job_id', 'ep_version', 'error', 'auth', 
                                        'worker_interface_version', 'start_time', 'start_time_compute',
                                        'arrival_time', 'finished_time', 'result_received_time',
                                        'result_sent_time', 'model_name', 'pending_duration',
                                        'preprocessing_duration', 'success', 'job_state',
                                        'queue_position', 'estimate', 'progress'
                                    }
                                    text_candidates = [v for k, v in result_data.items() 
                                                     if isinstance(v, str) and len(v) > 10 
                                                     and k.lower() not in metadata_keys]
                                    if text_candidates:
                                        text = text_candidates[0]
                                        log.debug(f"Extracted text from candidates: {text[:40]}")
                                        
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
                        else:
                            # Handle non-dict result_data
                            if isinstance(result_data, str) and result_data.strip():
                                text = result_data
                                log.debug(f"Using string result_data as text: {text[:40]}")
                                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                                metrics = {}
                                break
            
            # Execute with 3-minute timeout
            try:
                await asyncio.wait_for(process_api_response(), timeout=180)
            except asyncio.TimeoutError:
                log.error("AIME API timeout: No response within 180 seconds")
                # Use fallback response
                text = ""

        if not text:
            log.warning("No response text received from AIME API, using fallback")
            # Create a fallback response instead of raising an error
            text = "I apologize, but I cannot provide a response at this time due to technical issues with the AI service."
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            metrics = {}

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
            history=[],
            metrics=metrics,
        )

    async def ainvoke(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Async invoke entrypoint expected by evaluation adapters.

        Returns an object with a `.content` attribute containing the model output string.
        """
        config = config or {}
        history = config.get("history")
        # Pass through common generation kwargs if present
        gen_kwargs_keys = {
            "top_k",
            "top_p",
            "temperature",
            "max_gen_tokens",
            "wait_for_result",
            "json",
        }
        gen_kwargs = {k: v for k, v in config.items() if k in gen_kwargs_keys}
        resp = await self.achat(prompt, history=history, **gen_kwargs)
        # Normalize to an object exposing `.content`
        text = getattr(resp, "text", None)
        if text is None and hasattr(resp, "output"):
            # Fallback to BaseModelOutput
            try:
                text = getattr(resp.output, "content", None)
            except Exception:
                text = None

        if text is None:
            text = str(resp)

        class _SimpleResponse:
            def __init__(self, content: str):
                self.content = content
                self.text = content

        return _SimpleResponse(text)

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
            self._session_token = await self.model_api.do_api_login_async(
                user=self.email,
                api_key=self.api_key
            )
            self.model_api.client_session_auth_key = self._session_token
        # Create chat context with recommended AIME API format for streaming
        assistant_name = "GraphRAG Assistant"
        
        if history is None:
            chat_context = [
                {
                    "role": "system",
                    "content": (
                        f"You are a helpful, respectful and honest assistant named {assistant_name}. "
                        "You specialize in knowledge analysis, data interpretation, and providing insights from GraphRAG (Graph Retrieval-Augmented Generation) systems. "
                        "Always answer as helpfully as possible, while being safe. "
                        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                        "Please ensure that your responses are socially unbiased and positive in nature. "
                        "When analyzing data or providing insights, be precise and cite your sources when available. "
                        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                        "If you don't know the answer to a question, please don't share false information. "
                        "Focus on providing accurate, data-driven insights based on the information available to you."
                    )
                },
                {
                    "role": "user", 
                    "content": f"Hello, {assistant_name}."
                },
                {
                    "role": "assistant", 
                    "content": "Hello! I'm your GraphRAG Assistant, specialized in knowledge analysis and data interpretation. How can I help you analyze data or extract insights today?"
                }
            ]
        else:
            # For existing history, preserve the system prompt structure but ensure proper format
            chat_context = []
            system_prompt_added = False
            
            for msg in history:
                if isinstance(msg, dict):
                    chat_context.append(msg)
                    if msg.get("role") == "system":
                        system_prompt_added = True
                else:
                    chat_context.append({"role": "user", "content": str(msg)})
            
            # Add system prompt if not already present
            if not system_prompt_added:
                system_msg = {
                    "role": "system",
                    "content": (
                        f"You are a helpful, respectful and honest assistant named {assistant_name}. "
                        "You specialize in knowledge analysis, data interpretation, and providing insights from GraphRAG systems. "
                        "Always answer as helpfully as possible, while being safe and accurate."
                    )
                }
                chat_context.insert(0, system_msg)
        
        # Handle json parameter for streaming
        modified_prompt = prompt
        if kwargs.get("json", False):
            # Add JSON instruction to the prompt if it's not already there
            if "JSON" not in prompt.upper() and "json" not in prompt.lower():
                json_instruction = "\n\nIMPORTANT: Please respond ONLY with valid JSON format. Do not include any explanatory text before or after the JSON. Start your response with { and end with }."
                modified_prompt = prompt + json_instruction
        
        # Add the user prompt to chat_context as recommended by AIME API
        chat_context.append({"role": "user", "content": modified_prompt})
        chat_context_str = json.dumps(chat_context)
        
        # Extract model_parameters if provided (nested dict from GraphRAG)
        model_params = kwargs.get("model_parameters", {}) or {}
        
        params = {
            "chat_context": chat_context_str,  # Use chat_context instead of prompt_input
            "top_k": int(model_params.get("top_k", kwargs.get("top_k", 40))),
            "top_p": float(model_params.get("top_p", kwargs.get("top_p", 0.9))),
            "temperature": float(model_params.get("temperature", kwargs.get("temperature", 0.1))),
            "max_gen_tokens": int(model_params.get("max_gen_tokens", kwargs.get("max_gen_tokens", kwargs.get("max_tokens", 3500)))),
            "wait_for_result": False,
            "client_session_auth_key": self._session_token,
        }
        
        # Add chat_output_format if specified (to avoid <think> tags in gpt-oss responses)
        if self.chat_output_format:
            params["chat_output_format"] = self.chat_output_format
            log.info(f"AIME achat_stream - Setting chat_output_format to: {self.chat_output_format}")
        
        # Log the parameters being used for debugging
        log.info(f"AIME achat_stream - model_params from caller: {model_params}")
        log.info(f"AIME achat_stream - Final params: temperature={params['temperature']}, max_gen_tokens={params['max_gen_tokens']}, top_p={params['top_p']}")
        
        for k, v in kwargs.items():
            if k not in params and v is not None and k not in ['stream', 'model_parameters']:
                params[k] = v
        
        # Use retry mechanism for streaming as well
        max_stream_retries = 5
        base_retry_delay = 3.0
        
        for retry_attempt in range(max_stream_retries):
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
                
                # Successfully completed, break out of retry loop
                break
                        
            except json.JSONDecodeError as je:
                error_msg = f"Failed to decode JSON response: {str(je)}"
                log.error(f"{error_msg}. Response: {chunk if 'chunk' in locals() else 'No chunk data'}")
                if retry_attempt < max_stream_retries - 1:
                    backoff = base_retry_delay * (2**retry_attempt)
                    log.warning(f"JSON decode error on attempt {retry_attempt + 1}/{max_stream_retries}, retrying after {backoff:.2f}s")
                    await asyncio.sleep(backoff)
                    # Reset session
                    if hasattr(self.model_api, "session") and self.model_api.session:
                        await self.model_api.close_session()
                        self.model_api.session = None
                        self._session_token = None
                else:
                    raise ValueError(error_msg) from je
                    
            except Exception as e:
                error_str = str(e).lower()
                error_msg = f"Unexpected error in achat_stream: {str(e)}"
                log.error(error_msg)
                log.error(f"Exception type: {type(e).__name__}")
                try:
                    stack_trace = traceback.format_exc()
                    log.error(f"Stack trace: {stack_trace}")
                except Exception as traceback_error:
                    log.error(f"Failed to get stack trace: {traceback_error}")
                
                # Check if this is a retryable error
                is_retryable = (
                    "timeout" in error_str or
                    "network" in error_str or
                    "session" in error_str or
                    "connection" in error_str or
                    "disconnected" in error_str or
                    "server disconnected" in error_str or
                    "write request body" in error_str or
                    "no free queue slot" in error_str or
                    "queue slot" in error_str or
                    "status code: 400" in error_str or
                    "status code: 429" in error_str or
                    "status code: 503" in error_str or
                    isinstance(e, (
                        BrokenPipeError,
                        ConnectionResetError,
                        aiohttp.ClientOSError,
                        ServerDisconnectedError,
                        aiohttp.ServerDisconnectedError,
                    ))
                )
                
                if is_retryable and retry_attempt < max_stream_retries - 1:
                    backoff = base_retry_delay * (2**retry_attempt)
                    log.warning(f"Retryable error on attempt {retry_attempt + 1}/{max_stream_retries}, retrying after {backoff:.2f}s: {e}")
                    await asyncio.sleep(backoff)
                    # Reset session for retryable errors
                    if hasattr(self.model_api, "session") and self.model_api.session:
                        await self.model_api.close_session()
                        self.model_api.session = None
                        self._session_token = None
                elif not is_retryable:
                    # Non-retryable error, fail immediately
                    log.error(f"Non-retryable error: {e}")
                    raise ValueError(error_msg) from e
                else:
                    # Max retries reached
                    log.error(f"Max retries ({max_stream_retries}) reached. Error: {e}")
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
            self._session_token = await self.model_api.do_api_login_async(
                user=self.email,
                api_key=self.api_key
            )
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
