from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import asyncio
import logging
import json
import sys
import os

# Add the Evaluate directory to path for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.error_handling import handle_llm_api_error


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters."""
    
    @abstractmethod
    async def ainvoke(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Invoke LLM asynchronously."""
        pass
    
    @abstractmethod
    async def achat(self, prompt: str) -> Any:
        """Chat with LLM asynchronously."""
        pass
    
    async def close(self) -> None:
        """Close the adapter and cleanup resources. Override in subclasses if needed."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        await self.close()
        return False


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI-compatible LLMs."""
    
    def __init__(self, llm_instance: Any, logger: logging.Logger):
        self.llm = llm_instance
        self.logger = logger
    
    async def ainvoke(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Invoke LLM with prompt using a configurable timeout."""
        try:
            config = config or {}
            api_timeout = config.get("api_timeout", 180)
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt, config=config),
                timeout=api_timeout
            )
            return response
        except asyncio.TimeoutError:
            self.logger.error(f"OpenAI API timeout after {api_timeout} seconds")
            raise Exception(f"OpenAI API timeout: No response within {api_timeout} seconds")
        except Exception as e:
            error_msg = handle_llm_api_error(e, self.logger)
            raise Exception(error_msg)
    
    async def achat(self, prompt: str) -> Any:
        """Chat with LLM with a configurable timeout."""
        try:
            # Keep compatibility with callers that don't pass config here.
            api_timeout = 180
            response = await asyncio.wait_for(
                self.llm.achat(prompt),
                timeout=api_timeout
            )
            return response
        except asyncio.TimeoutError:
            self.logger.error(f"OpenAI API chat timeout after {api_timeout} seconds")
            raise Exception(f"OpenAI API chat timeout: No response within {api_timeout} seconds")
        except Exception as e:
            error_msg = handle_llm_api_error(e, self.logger)
            raise Exception(error_msg)


class AIMEAPIAdapter(LLMAdapter):
    """Adapter for AIME API."""
    
    # Default LLM parameters for reproducible benchmarking
    # These can be overridden via constructor or config dict in ainvoke()
    DEFAULT_TEMPERATURE = 0.0  # Deterministic for benchmarking
    DEFAULT_MAX_TOKENS = 3500
    DEFAULT_TOP_P = 1.0        # No filtering for deterministic mode
    DEFAULT_TOP_K = 40
    DEFAULT_API_TIMEOUT = 180
    
    def __init__(self, model_api: Any, logger: logging.Logger, user: Optional[str] = None, 
                 api_key: Optional[str] = None, default_params: Optional[Dict[str, Any]] = None):
        self.model_api = model_api
        self.logger = logger
        self._logged_in = False
        
        # Store default LLM parameters for reproducible benchmarking
        # These are used as defaults in ainvoke() unless overridden by config
        self._default_params = default_params or {}
        self._temperature = self._default_params.get('temperature', self.DEFAULT_TEMPERATURE)
        self._max_tokens = self._default_params.get('max_tokens', self.DEFAULT_MAX_TOKENS)
        self._top_p = self._default_params.get('top_p', self.DEFAULT_TOP_P)
        self._top_k = self._default_params.get('top_k', self.DEFAULT_TOP_K)
        self._api_timeout = int(self._default_params.get('api_timeout', self.DEFAULT_API_TIMEOUT))
        
        # Store credentials for login - check multiple sources in order of priority:
        # 1. Explicit parameters passed to constructor
        # 2. Custom _aime_user/_aime_api_key attributes (set by model_manager)
        # 3. Standard model_api attributes (user, api_key)
        self._user = user or getattr(model_api, '_aime_user', None) or getattr(model_api, 'user', None)
        self._api_key = api_key or getattr(model_api, '_aime_api_key', None) or getattr(model_api, 'api_key', None)
    
    async def _ensure_login(self):
        """Ensure the API client is logged in with proper session management."""
        if not self._logged_in:
            try:
                self.logger.info("=" * 80)
                self.logger.info("AIME API LOGIN REQUEST")
                self.logger.info("=" * 80)
                self.logger.info(f"API URL: {getattr(self.model_api, 'api_server', 'N/A')}")
                self.logger.info(f"Model Name: {getattr(self.model_api, 'endpoint_name', 'N/A')}")
                self.logger.info(f"User: {self._user or 'N/A'}")
                self.logger.info(f"API Key Present: {bool(self._api_key)}")
                
                # Validate credentials before attempting login
                if not self._user:
                    raise ValueError("Cannot login: user credential is missing. Check that the AIME config is properly set in settings.yaml (email/user field)")
                if not self._api_key:
                    raise ValueError("Cannot login: api_key credential is missing. Check that the AIME config is properly set in settings.yaml (api_key field)")
                
                # Ensure we have a proper session before login
                import aiohttp
                if not hasattr(self.model_api, 'session') or self.model_api.session is None:
                    # Create a new session with proper connector settings
                    connector = aiohttp.TCPConnector(
                        limit=10,
                        limit_per_host=5,
                        keepalive_timeout=30,
                        enable_cleanup_closed=True
                    )
                    timeout = aiohttp.ClientTimeout(total=300, connect=30)
                    self.model_api.session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout
                    )
                elif hasattr(self.model_api.session, 'closed') and self.model_api.session.closed:
                    # Session was closed, create a new one
                    connector = aiohttp.TCPConnector(
                        limit=10,
                        limit_per_host=5,
                        keepalive_timeout=30,
                        enable_cleanup_closed=True
                    )
                    timeout = aiohttp.ClientTimeout(total=300, connect=30)
                    self.model_api.session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout
                    )
                
                # Use async login for async operations - pass user and api_key explicitly
                session_token = await self.model_api.do_api_login_async(
                    user=self._user,
                    api_key=self._api_key
                )
                self.model_api.client_session_auth_key = session_token
                self._logged_in = True
                self.logger.info("✓ AIME API login successful")
                self.logger.info("=" * 80)
            except Exception as e:
                self.logger.error(f"✗ AIME API login failed: {str(e)}")
                self.logger.info("=" * 80)
                self._logged_in = False
                # Clean up failed session
                if hasattr(self.model_api, 'session') and self.model_api.session:
                    try:
                        await self.model_api.session.close()
                    except:
                        pass
                    self.model_api.session = None
                raise
    
    async def ainvoke(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Invoke AIME API with prompt using the recommended chat_context format."""
        max_retries = 5  # Increased from 3 to handle queue slot issues
        retry_delay = 3.0  # Increased from 1.0 to give more time between retries
        
        for attempt in range(max_retries):
            try:
                await self._ensure_login()
                
                config = config or {}

                # Optional JSON-mode prompt constraint (AIME server does not expose an OpenAI-style response_format)
                if config.get("json") is True:
                    prompt = (
                        prompt
                        + "\n\nIMPORTANT: Respond ONLY with valid JSON. Do not include any other text."
                    )

                # Create chat context using the recommended AIME API format
                assistant_name = "GraphRAG Assistant"
                chat_context = [
                    {
                        "role": "system",
                        "content": (
                            f"You are a helpful, respectful and honest assistant named {assistant_name}. "
                            "You specialize in knowledge analysis, data interpretation, and providing insights from GraphRAG systems. "
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
                        "content": "Hello! I'm your GraphRAG Assistant, specialized in knowledge analysis and data interpretation. How can I help you today?"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                
                # Build parameters according to new AIME API specification
                # Use instance defaults for reproducible benchmarking (temperature=0 by default)
                params = {
                    "chat_context": json.dumps(chat_context),  # Use chat_context instead of prompt_input
                    "wait_for_result": True,
                    "top_k": self._top_k,
                    "top_p": self._top_p,
                    "temperature": self._temperature,  # Default 0.0 for reproducibility
                    "max_gen_tokens": self._max_tokens
                }
                
                # Add optional parameters if provided in config
                if config:
                    if 'temperature' in config:
                        params['temperature'] = config['temperature']
                    if 'max_tokens' in config:
                        params['max_gen_tokens'] = config['max_tokens']
                    if 'top_p' in config:
                        params['top_p'] = config['top_p']
                    if 'top_k' in config:
                        params['top_k'] = config['top_k']
                
                # Log the complete request details
                self.logger.info("=" * 80)
                self.logger.info(f"AIME API REQUEST (Attempt {attempt + 1}/{max_retries})")
                self.logger.info("=" * 80)
                self.logger.info(f"API URL: {getattr(self.model_api, 'api_url', 'N/A')}")
                self.logger.info(f"Model Name: {getattr(self.model_api, 'model_name', 'N/A')}")
                self.logger.info(f"Request Parameters:")
                
                # Log parameters in a readable format
                for key, value in params.items():
                    if key == 'chat_context':
                        # Parse and show chat context structure
                        try:
                            context_data = json.loads(value)
                            self.logger.info(f"  - {key}: {len(context_data)} messages")
                            for i, msg in enumerate(context_data):
                                role = msg.get('role', 'unknown')
                                content_preview = msg.get('content', '')[:100] + "..." if len(msg.get('content', '')) > 100 else msg.get('content', '')
                                self.logger.info(f"    [{i+1}] {role}: {content_preview}")
                        except:
                            self.logger.info(f"  - {key}: [chat context string]")
                        self.logger.info(f"  - chat_context_length: {len(value)} characters")
                    else:
                        self.logger.info(f"  - {key}: {value}")
                
                self.logger.info("-" * 80)
                self.logger.info("Full Request Parameters (JSON):")
                try:
                    # Create a copy with truncated chat_context for JSON logging
                    params_log = params.copy()
                    if 'chat_context' in params_log and len(params_log['chat_context']) > 500:
                        params_log['chat_context'] = params_log['chat_context'][:500] + "... [truncated]"
                    self.logger.info(json.dumps(params_log, indent=2))
                except Exception as json_err:
                    self.logger.warning(f"Could not serialize params to JSON: {json_err}")
                self.logger.info("=" * 80)
                
                output_generator = self.model_api.get_api_request_generator(params)
                
                answer = None
                api_timeout = int(config.get("api_timeout", self._api_timeout))
                try:
                    async def process_api_response():
                        """Process API response with proper error handling."""
                        nonlocal answer
                        async for progress in output_generator:
                            # Only handle error and final response, skip progress update prints
                            if isinstance(progress, dict) and progress.get('job_state') == 'error':
                                error_description = progress.get('error_description', 'Unknown API error')
                                error_data = progress.get('error_data', {})
                                self.logger.error("=" * 80)
                                self.logger.error("AIME API ERROR RESPONSE")
                                self.logger.error("=" * 80)
                                self.logger.error(f"Error Description: {error_description}")
                                self.logger.error(f"Error Data: {json.dumps(error_data, indent=2)}")
                                self.logger.error(f"Full Progress Dict: {json.dumps(progress, indent=2)}")
                                self.logger.error("=" * 80)
                                raise Exception(f"AIME API error: {error_description}")
                            # Extract result when job is done
                            if isinstance(progress, dict) and progress.get('job_state') == 'done':
                                result_data = progress.get('result_data', {})
                                self.logger.info("=" * 80)
                                self.logger.info("AIME API RESPONSE (Job Done)")
                                self.logger.info("=" * 80)
                                self.logger.info(f"Result Data Keys: {list(result_data.keys())}")
                                
                                # IMPORTANT: Check for error in result_data first
                                # The API returns 'error' key (no 'text') when request fails (e.g., context too large)
                                if result_data.get('error'):
                                    error_msg = result_data.get('error')
                                    self.logger.error(f"✗ AIME API returned error in result_data: {error_msg}")
                                    self.logger.error("=" * 80)
                                    # Raise with specific error message from API
                                    raise Exception(f"AIME API error: {error_msg}")
                                
                                answer = (
                                    result_data.get('text') or 
                                    result_data.get('output') or 
                                    result_data.get('generated_text')
                                )
                                if not answer and result_data:
                                    for key in ['response', 'completion', 'result']:
                                        if key in result_data and isinstance(result_data[key], str):
                                            answer = result_data[key]
                                            self.logger.info(f"Extracted answer from key: {key}")
                                            break
                                if answer:
                                    answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
                                    self.logger.info(f"✓ Answer received ({len(answer)} characters)")
                                    self.logger.info(f"Answer preview: {answer_preview}")
                                    self.logger.info("=" * 80)
                                    return  # Exit the function successfully
                                else:
                                    self.logger.error("✗ No answer found in result_data")
                                    self.logger.info("=" * 80)
                    
                    # Execute the API call with timeout
                    await asyncio.wait_for(process_api_response(), timeout=api_timeout)
                    
                except asyncio.TimeoutError:
                    self.logger.error("=" * 80)
                    self.logger.error(f"AIME API TIMEOUT ERROR (Attempt {attempt + 1}/{max_retries})")
                    self.logger.error("=" * 80)
                    self.logger.error(f"API call exceeded {api_timeout} seconds timeout")
                    self.logger.error("=" * 80)
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Timeout detected, retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        raise Exception(f"AIME API timeout after {max_retries} attempts: No response within {api_timeout} seconds")
                
                if not answer:
                    self.logger.error("=" * 80)
                    self.logger.error("AIME API ERROR: No Response")
                    self.logger.error("=" * 80)
                    self.logger.error("No response text extracted from any progress update")
                    self.logger.error("=" * 80)
                    raise Exception("No response received from AIME API - job completed but no text extracted")
                
                # Create a response-like object to maintain compatibility with other adapters
                class MockResponse:
                    def __init__(self, content: str):
                        self.content = content
                
                self.logger.info("✓ Request completed successfully")
                return MockResponse(answer)
                
            except asyncio.TimeoutError:
                # Already handled above, re-raise to be caught by outer exception handlers
                raise
                
            except (BrokenPipeError, ConnectionError, ConnectionRefusedError, 
                    AttributeError, OSError) as e:
                error_str = str(e)
                self.logger.error("=" * 80)
                self.logger.error(f"AIME API CONNECTION ERROR (Attempt {attempt + 1}/{max_retries})")
                self.logger.error("=" * 80)
                self.logger.error(f"Error Type: {type(e).__name__}")
                self.logger.error(f"Error Message: {error_str}")
                self.logger.error("=" * 80)
                
                # Connection errors are always retryable
                if attempt < max_retries - 1:
                    self.logger.warning(f"Connection error detected, retrying in {retry_delay} seconds...")
                    # Reset login state on connection errors to force re-login
                    self._logged_in = False
                    
                    # Clean up session properly
                    await self._cleanup_session()
                    
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    error_msg = handle_llm_api_error(e, self.logger)
                    raise Exception(error_msg)
            
            except Exception as e:
                error_str = str(e)
                self.logger.error("=" * 80)
                self.logger.error(f"AIME API EXCEPTION (Attempt {attempt + 1}/{max_retries})")
                self.logger.error("=" * 80)
                self.logger.error(f"Exception Type: {type(e).__name__}")
                self.logger.error(f"Exception Message: {error_str}")
                
                # Try to get more details from the exception
                if hasattr(e, '__dict__'):
                    self.logger.error(f"Exception Attributes: {e.__dict__}")
                
                import traceback
                self.logger.error("Traceback:")
                self.logger.error(traceback.format_exc())
                self.logger.error("=" * 80)
                
                # Check if this is a retryable error
                is_retryable = (
                    "timeout" in error_str.lower() or
                    "network" in error_str.lower() or
                    "session" in error_str.lower() or
                    "lost connection" in error_str.lower() or
                    "nonetype" in error_str.lower() or
                    "attribute" in error_str.lower() or
                    "no free queue slot" in error_str.lower() or
                    "queue slot" in error_str.lower() or
                    "status code: 400" in error_str.lower()
                )
                
                if is_retryable and attempt < max_retries - 1:
                    self.logger.warning(f"Retryable error detected, retrying in {retry_delay} seconds...")
                    # Reset login state on retryable errors
                    self._logged_in = False
                    await self._cleanup_session()
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    # Either not retryable or we've exhausted retries
                    error_msg = handle_llm_api_error(e, self.logger)
                    raise Exception(error_msg)
        
        # If we get here, all retries failed
        raise Exception(f"AIME API request failed after {max_retries} attempts")

    async def _cleanup_session(self):
        """Clean up the current session properly."""
        try:
            if hasattr(self.model_api, 'session') and self.model_api.session:
                if hasattr(self.model_api.session, 'close'):
                    await self.model_api.session.close()
                elif hasattr(self.model_api, 'close_session'):
                    await self.model_api.close_session()
        except Exception as e:
            self.logger.warning(f"Error during session cleanup: {e}")
        finally:
            self.model_api.session = None
            if hasattr(self.model_api, 'client_session_auth_key'):
                self.model_api.client_session_auth_key = None
    
    async def close(self):
        """Close the adapter and cleanup resources."""
        await self._cleanup_session()
        self._logged_in = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup session."""
        await self.close()
        return False
    
    async def achat(self, prompt: str) -> Any:
        """Chat with AIME API."""
        return await self.ainvoke(prompt)


class LangChainAdapter(LLMAdapter):
    """Adapter for LangChain LLMs."""
    
    def __init__(self, llm_instance: Any, logger: logging.Logger):
        self.llm = llm_instance
        self.logger = logger
    
    async def ainvoke(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Invoke LangChain LLM."""
        try:
            response = await self.llm.ainvoke(prompt, config=config or {})
            return response
        except Exception as e:
            error_msg = handle_llm_api_error(e, self.logger)
            raise Exception(error_msg)
    
    async def achat(self, prompt: str) -> Any:
        """Chat with LangChain LLM."""
        try:
            response = await self.llm.achat(prompt)
            return response
        except Exception as e:
            error_msg = handle_llm_api_error(e, self.logger)
            raise Exception(error_msg)


class LLMAdapterFactory:
    """Factory for creating appropriate LLM adapters."""
    
    @staticmethod
    def create_adapter(llm_instance: Any, logger: logging.Logger, 
                      adapter_type: str = "auto",
                      user: Optional[str] = None,
                      api_key: Optional[str] = None,
                      default_params: Optional[Dict[str, Any]] = None) -> LLMAdapter:
        """Create LLM adapter based on instance type or specified type.
        
        Args:
            llm_instance: The LLM instance to wrap
            logger: Logger instance
            adapter_type: Type of adapter ("auto", "aime", "openai", "langchain")
            user: User credential for AIME API
            api_key: API key for AIME API
            default_params: Default LLM parameters for reproducible benchmarking
                           (temperature, max_tokens, top_p, top_k)
        """
        
        if adapter_type == "aime":
            return AIMEAPIAdapter(llm_instance, logger, user=user, api_key=api_key, 
                                 default_params=default_params)
        elif adapter_type == "openai":
            return OpenAIAdapter(llm_instance, logger)
        elif adapter_type == "langchain":
            return LangChainAdapter(llm_instance, logger)
        else:
            # Auto-detect based on instance attributes
            if hasattr(llm_instance, 'get_api_request_generator'):
                return AIMEAPIAdapter(llm_instance, logger, user=user, api_key=api_key,
                                     default_params=default_params)
            elif hasattr(llm_instance, 'ainvoke') and hasattr(llm_instance, 'achat'):
                return LangChainAdapter(llm_instance, logger)
            else:
                # Default to OpenAI adapter
                return OpenAIAdapter(llm_instance, logger) 