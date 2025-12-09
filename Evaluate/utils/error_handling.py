from contextlib import contextmanager
from typing import Any, Optional
import logging
import traceback


class MetricError(Exception):
    """Base exception for metric computation errors."""
    pass


class QueryExecutionError(Exception):
    """Exception for query execution errors."""
    pass


class ConfigurationError(Exception):
    """Exception for configuration errors."""
    pass


class SafeMetricComputation:
    """Context manager for safe metric computation that properly propagates fallback values."""
    
    def __init__(self, metric_name: str, logger: logging.Logger, 
                 fallback_value: Any = None, question: str = "", method: str = ""):
        self.metric_name = metric_name
        self.logger = logger
        self.fallback_value = fallback_value
        self.question = question
        self.method = method
        self.exception_occurred = False
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_msg = f"Metric {self.metric_name} failed: {str(exc_val)}"
            if self.question and self.method:
                error_msg += f" (Question: {self.question[:50]}..., Method: {self.method})"
            
            self.logger.warning(error_msg)
            self.exception_occurred = True
            
            if self.fallback_value is not None:
                # Suppress the exception - caller should check exception_occurred
                return True
            # Let the exception propagate
            return False
        return False


def safe_metric_computation(metric_name: str, logger: logging.Logger, 
                          fallback_value: Any = None, question: str = "", method: str = ""):
    """Factory function for safe metric computation context manager.
    
    Usage:
        with safe_metric_computation('my_metric', logger, fallback_value=0.0):
            # Your metric computation code
            result = compute_something()
            if result is None:
                return fallback_value  # Explicit fallback
            return result
    
    Note: When an exception occurs and fallback_value is provided, the exception
    is suppressed. The calling code should return the fallback_value explicitly
    in the normal flow or after checking for errors.
    """
    return SafeMetricComputation(metric_name, logger, fallback_value, question, method)


@contextmanager
def safe_query_execution(method: str, logger: logging.Logger, question: str = ""):
    """Context manager for safe query execution."""
    try:
        yield
    except Exception as e:
        error_msg = f"Query execution failed for {method}: {str(e)}"
        if question:
            error_msg += f" (Question: {question[:50]}...)"
        
        logger.error(error_msg)
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        raise QueryExecutionError(f"Query execution failed for {method}: {str(e)}")


def handle_llm_api_error(error: Exception, logger: logging.Logger, 
                        question: str = "", method: str = "") -> str:
    """Handle LLM API errors and return appropriate error message."""
    error_msg = f"LLM API error: {str(error)}"
    
    if "rate limit" in str(error).lower():
        error_msg = "Rate limit exceeded. Please try again later."
    elif "authentication" in str(error).lower():
        error_msg = "Authentication failed. Please check your API credentials."
    elif "timeout" in str(error).lower():
        error_msg = "Request timed out. Please try again."
    elif "model not found" in str(error).lower():
        error_msg = "Model not found. Please check your model configuration."
    
    logger.error(f"{error_msg} (Question: {question[:50]}..., Method: {method})")
    return error_msg


def validate_input_data(data: Any, required_fields: list, logger: logging.Logger) -> bool:
    """Validate input data has required fields."""
    if not isinstance(data, dict):
        logger.error(f"Input data must be a dictionary, got {type(data)}")
        return False
    
    missing_fields = []
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
    
    if missing_fields:
        logger.error(f"Missing required fields: {missing_fields}")
        return False
    
    return True


def sanitize_text(text: str, max_length: int = None, truncate: bool = False) -> str:
    """Sanitize text for safe processing.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum length (only applies if truncate=True). Default: 10000
        truncate: If True, truncate text to max_length. Default: False (preserve full text)
                  Set to False for ground truth/answer fields to preserve metric accuracy.
    
    Returns:
        Sanitized text string
    """
    if not isinstance(text, str):
        return ""
    
    # Remove null bytes and other problematic characters
    text = text.replace('\x00', '')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Only truncate if explicitly requested (preserves ground truth accuracy)
    if truncate:
        if max_length is None:
            max_length = 10000  # Default max if truncation requested
        if len(text) > max_length:
            text = text[:max_length] + "..."
    
    return text


def parse_json_response(response_text: str, logger: logging.Logger, 
                       context: str = "", raise_on_failure: bool = True) -> Any:
    """Parse JSON response from LLM, handling common formatting issues.
    
    Args:
        response_text: The text response from the LLM
        logger: Logger instance for warnings
        context: Context string for error messages
        raise_on_failure: If True, raise exception on parse failure. If False, return empty list/dict.
        
    Returns:
        Parsed JSON object (dict, list, etc.). Returns [] on failure if raise_on_failure=False.
        
    Raises:
        json.JSONDecodeError: If JSON parsing fails after cleanup attempts (only if raise_on_failure=True)
    """
    import json
    import re
    
    if not response_text:
        logger.warning(f"Empty response text for {context}")
        return []
    
    # Clean up response text
    cleaned_text = response_text.strip()
    
    # Store original for JSON extraction from thinking blocks
    original_text = cleaned_text
    
    # Remove <think>...</think> tags (chain-of-thought reasoning) - common in reasoning models
    # Use a more aggressive pattern that handles nested tags and various formats
    think_patterns = [
        r'<think>[\s\S]*?</think>',  # Standard <think>...</think>
        r'<thinking>[\s\S]*?</thinking>',  # Alternative <thinking>...</thinking>
        r'<thought>[\s\S]*?</thought>',  # Alternative <thought>...</thought>
        r'<reasoning>[\s\S]*?</reasoning>',  # Alternative <reasoning>...</reasoning>
    ]
    for pattern in think_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE | re.DOTALL).strip()
    
    # Handle UNCLOSED think tags (common issue with reasoning models)
    # If text starts with <think> but has no closing tag, try to extract JSON after the tag content
    unclosed_think_patterns = [
        r'^<think>',
        r'^<thinking>',
        r'^<thought>',
        r'^<reasoning>',
    ]
    for pattern in unclosed_think_patterns:
        if re.match(pattern, cleaned_text, re.IGNORECASE):
            # Find the first [ or { which likely starts the JSON
            json_array_start = cleaned_text.find('[')
            json_obj_start = cleaned_text.find('{')
            
            # Determine which comes first (if both exist)
            if json_array_start != -1 and json_obj_start != -1:
                json_start = min(json_array_start, json_obj_start)
            elif json_array_start != -1:
                json_start = json_array_start
            elif json_obj_start != -1:
                json_start = json_obj_start
            else:
                json_start = -1
            
            if json_start != -1:
                cleaned_text = cleaned_text[json_start:]
                break
    
    # If text is empty after removing think tags, try to extract JSON from original
    if not cleaned_text:
        # Try to find JSON in the original text (inside think tags)
        json_array_start = original_text.find('[')
        json_obj_start = original_text.find('{')
        
        if json_array_start != -1 or json_obj_start != -1:
            if json_array_start != -1 and json_obj_start != -1:
                json_start = min(json_array_start, json_obj_start)
            elif json_array_start != -1:
                json_start = json_array_start
            else:
                json_start = json_obj_start
            
            cleaned_text = original_text[json_start:]
        else:
            logger.warning(f"Empty text after removing think tags for {context}")
            return []
    
    def try_parse_json(text: str) -> Any:
        """Attempt to parse JSON with various fixups."""
        # Direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Fix unescaped newlines in strings
        # This is a common issue with LLM responses
        def escape_newlines_in_strings(s):
            result = []
            in_string = False
            escape_next = False
            for char in s:
                if escape_next:
                    result.append(char)
                    escape_next = False
                    continue
                if char == '\\':
                    result.append(char)
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    result.append(char)
                    continue
                if in_string and char == '\n':
                    result.append('\\n')
                    continue
                if in_string and char == '\r':
                    result.append('\\r')
                    continue
                if in_string and char == '\t':
                    result.append('\\t')
                    continue
                result.append(char)
            return ''.join(result)
        
        try:
            fixed = escape_newlines_in_strings(text)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Remove trailing commas
        try:
            fixed = re.sub(r',\s*([\]}])', r'\1', text)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Remove single-line comments
        try:
            fixed = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Combine fixes
        try:
            fixed = escape_newlines_in_strings(text)
            fixed = re.sub(r',\s*([\]}])', r'\1', fixed)
            fixed = re.sub(r'//.*$', '', fixed, flags=re.MULTILINE)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        return None
    
    # Try direct parsing first
    result = try_parse_json(cleaned_text)
    if result is not None:
        return result
    
    # Try to extract JSON from markdown code blocks (```json ... ``` or ``` ... ```)
    # Handle both ```json and just ```
    patterns_for_code_blocks = [
        r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
    ]
    
    for pattern in patterns_for_code_blocks:
        json_match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
        if json_match:
            extracted = json_match.group(1).strip()
            result = try_parse_json(extracted)
            if result is not None:
                return result
    
    # Also try to find code blocks in original text (in case JSON is in thinking block)
    for pattern in patterns_for_code_blocks:
        json_match = re.search(pattern, original_text, re.DOTALL | re.IGNORECASE)
        if json_match:
            extracted = json_match.group(1).strip()
            result = try_parse_json(extracted)
            if result is not None:
                return result
    
    # Try to find JSON array in the text (greedy match for nested structures)
    # Look for array starting with [ and ending with ]
    start_idx = cleaned_text.find('[')
    if start_idx != -1:
        bracket_count = 0
        end_idx = start_idx
        for i, char in enumerate(cleaned_text[start_idx:], start_idx):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > start_idx:
            potential_json = cleaned_text[start_idx:end_idx]
            result = try_parse_json(potential_json)
            if result is not None:
                return result
    
    # Try to find JSON object in the text (greedy match for nested structures)
    start_idx = cleaned_text.find('{')
    if start_idx != -1:
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(cleaned_text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > start_idx:
            potential_json = cleaned_text[start_idx:end_idx]
            result = try_parse_json(potential_json)
            if result is not None:
                return result
    
    # Last resort: try to find any JSON-like structure using simple patterns
    # This is more aggressive and may have false positives
    for pattern in [r'\[.*\]', r'\{.*\}']:
        json_match = re.search(pattern, cleaned_text, re.DOTALL)
        if json_match:
            result = try_parse_json(json_match.group(0))
            if result is not None:
                return result
    
    # Final fallback: search in original text (handles JSON inside thinking blocks)
    for text_to_search in [original_text]:
        # Try to find JSON array
        start_idx = text_to_search.find('[')
        if start_idx != -1:
            bracket_count = 0
            end_idx = start_idx
            for i, char in enumerate(text_to_search[start_idx:], start_idx):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                potential_json = text_to_search[start_idx:end_idx]
                result = try_parse_json(potential_json)
                if result is not None:
                    return result
        
        # Try to find JSON object
        start_idx = text_to_search.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(text_to_search[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                potential_json = text_to_search[start_idx:end_idx]
                result = try_parse_json(potential_json)
                if result is not None:
                    return result
    
    # If all parsing attempts fail, log warning and either raise or return default
    # Check if response appears to be truncated (common with reasoning models that use too many tokens on <think>)
    truncation_indicators = [
        cleaned_text.count('{') != cleaned_text.count('}'),  # Unbalanced braces
        cleaned_text.count('[') != cleaned_text.count(']'),  # Unbalanced brackets
        cleaned_text.endswith('...'),  # Explicit truncation marker
        cleaned_text.endswith(','),    # Truncated in middle of array/object
        original_text.find('</think>') == -1 and original_text.find('<think>') != -1,  # Unclosed think tag
    ]
    
    is_truncated = any(truncation_indicators)
    
    if is_truncated:
        logger.warning(f"JSON response appears truncated for {context}. Response may have exceeded token limit.")
        logger.debug(f"Truncated response (last 200 chars): ...{cleaned_text[-200:] if len(cleaned_text) > 200 else cleaned_text}")
    else:
        logger.warning(f"Failed to parse JSON for {context}: {cleaned_text[:200]}...")
    
    if raise_on_failure:
        raise json.JSONDecodeError(f"Could not parse JSON from response for {context}", 
                                  response_text, 0)
    else:
        # Return appropriate default based on what we detected
        if cleaned_text.strip().startswith('['):
            return []
        else:
            return {}


def extract_first_number(text: str, logger: logging.Logger, 
                        context: str = "") -> Optional[float]:
    """Extract the first number from a text response.
    
    Args:
        text: The text containing a number
        logger: Logger instance for warnings
        context: Context string for error messages
        
    Returns:
        The first number found, or None if no number is found
    """
    import re
    
    if not text:
        logger.warning(f"Empty text for extracting number in {context}")
        return None
    
    # Try to find a number (integer or float)
    number_match = re.search(r'-?\d+\.?\d*', text)
    
    if number_match:
        try:
            return float(number_match.group(0))
        except ValueError:
            logger.warning(f"Found number pattern but couldn't convert to float in {context}: {number_match.group(0)}")
            return None
    
    logger.warning(f"No number found in text for {context}: {text[:100]}...")
    return None 