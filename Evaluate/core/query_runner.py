from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Dict, Optional
import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path
from copy import deepcopy
import pandas as pd

# Import GraphRAG components
try:
    from graphrag.api.query import local_search, global_search, basic_search
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    logging.warning("GraphRAG not available, some query runners may not work")

# Import AIME API components
try:
    from aime_api_client_interface.model_api import ModelAPI
    AIME_AVAILABLE = True
except ImportError:
    AIME_AVAILABLE = False
    logging.warning("AIME API not available, some query runners may not work")


def clean_think_tags(text: str) -> str:
    """Remove <think>, <thinking>, <thought>, <reasoning> tags from LLM responses.
    
    Reasoning models like GPT-Oss produce responses with <think>...</think> tags
    containing chain-of-thought reasoning. This function strips those tags to
    extract the actual answer.
    
    Args:
        text: The raw LLM response text
        
    Returns:
        Cleaned text with thinking tags removed
    """
    if not text or not isinstance(text, str):
        return text or ""
    
    cleaned = text.strip()
    
    # Patterns for various thinking tag formats
    think_patterns = [
        r'<think>[\s\S]*?</think>',      # Standard <think>...</think>
        r'<thinking>[\s\S]*?</thinking>',  # Alternative <thinking>...</thinking>
        r'<thought>[\s\S]*?</thought>',    # Alternative <thought>...</thought>
        r'<reasoning>[\s\S]*?</reasoning>',  # Alternative <reasoning>...</reasoning>
    ]
    
    for pattern in think_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Handle UNCLOSED think tags - find tags without matching closing tags
    # Strategy: If opening tag exists without closing tag, look for answer after newlines
    # or remove the tag and everything up to first substantial content
    unclosed_tags = [
        ('think', '<think>', '</think>'),
        ('thinking', '<thinking>', '</thinking>'),
        ('thought', '<thought>', '</thought>'),
        ('reasoning', '<reasoning>', '</reasoning>'),
    ]
    
    for tag_name, open_tag, close_tag in unclosed_tags:
        # Check if opening tag exists but closing tag doesn't
        if open_tag.lower() in cleaned.lower():
            # Check if there's no closing tag after the opening tag
            open_pos = cleaned.lower().find(open_tag.lower())
            close_pos = cleaned.lower().find(close_tag.lower(), open_pos)
            
            if close_pos == -1:  # No closing tag found
                # For unclosed tags, typically the answer comes after the reasoning
                # Look for patterns like: <think>reasoning\nActual Answer
                remaining = cleaned[open_pos + len(open_tag):]
                
                # Try to find where the actual answer starts (after reasoning content)
                # Common pattern: reasoning text followed by answer after newline
                lines = remaining.split('\n')
                if len(lines) > 1:
                    # Heuristic: Last non-empty line is likely the answer
                    for line in reversed(lines):
                        if line.strip():
                            cleaned = line.strip()
                            break
                else:
                    # Single line - just use content after tag
                    cleaned = remaining.strip()
                break
    
    return cleaned.strip()


def extract_context_texts_from_graphrag(context_data: Any, max_text_length: int = 50000, 
                                        logger: Optional[logging.Logger] = None) -> List[str]:
    """Extract full text content from GraphRAG context data for RAGAS evaluation.
    
    GraphRAG returns context as a dictionary with keys like 'reports', 'entities', 
    'relationships', 'claims', 'sources' containing DataFrames. This function
    extracts the actual text content from these structures to provide meaningful
    context for RAGAS metrics evaluation.
    
    Args:
        context_data: The context data returned by GraphRAG (dict or list)
        max_text_length: Maximum length per individual text to include (for sources)
        logger: Optional logger for warning messages when context extraction fails
        
    Returns:
        List of context text strings for RAGAS evaluation
    """
    contexts = []
    
    if context_data is None:
        return contexts
    
    if isinstance(context_data, dict):
        # Handle GraphRAG local_search and basic_search context formats
        # Priority order: sources (full text), reports, entities
        # Note: basic_search returns 'Sources' (capital S), local_search returns 'sources' (lowercase)
        
        # Create a case-insensitive lookup helper
        context_keys_lower = {k.lower(): k for k in context_data.keys()}
        
        # Extract from 'sources' or 'Sources' - these contain the original text units
        sources_key = context_keys_lower.get('sources')
        if sources_key:
            sources = context_data[sources_key]
            if isinstance(sources, pd.DataFrame) and 'text' in sources.columns:
                for _, row in sources.iterrows():
                    text = str(row.get('text', ''))
                    if text and len(text) > 10:  # Skip very short texts
                        # Include source_id if available for reference
                        source_id = row.get('source_id', row.get('id', ''))
                        if source_id:
                            contexts.append(f"[Source {source_id}]: {text[:max_text_length]}")
                        else:
                            contexts.append(text[:max_text_length])
        
        # Extract from 'reports' or 'Reports' - community reports with summaries
        reports_key = context_keys_lower.get('reports')
        if reports_key:
            reports = context_data[reports_key]
            if isinstance(reports, pd.DataFrame) and 'content' in reports.columns:
                for _, row in reports.iterrows():
                    content = str(row.get('content', ''))
                    if content and len(content) > 10:
                        title = row.get('title', '')
                        if title:
                            contexts.append(f"[Report: {title}]: {content[:max_text_length]}")
                        else:
                            contexts.append(content[:max_text_length])
        
        # Extract from 'entities' or 'Entities' - entity descriptions can provide useful context
        entities_key = context_keys_lower.get('entities')
        if entities_key:
            entities = context_data[entities_key]
            if isinstance(entities, pd.DataFrame) and 'description' in entities.columns:
                entity_descriptions = []
                for _, row in entities.iterrows():
                    desc = str(row.get('description', ''))
                    entity = row.get('entity', row.get('title', ''))
                    if desc and entity and len(desc) > 10:
                        entity_descriptions.append(f"{entity}: {desc}")
                if entity_descriptions:
                    contexts.append("[Entities]:\n" + "\n".join(entity_descriptions[:20]))  # Limit to 20 entities
        
        # Extract from 'relationships' or 'Relationships' - relationship descriptions
        relationships_key = context_keys_lower.get('relationships')
        if relationships_key:
            relationships = context_data[relationships_key]
            if isinstance(relationships, pd.DataFrame) and 'description' in relationships.columns:
                rel_descriptions = []
                for _, row in relationships.iterrows():
                    desc = str(row.get('description', ''))
                    source = row.get('source', '')
                    target = row.get('target', '')
                    if desc and source and target and len(desc) > 5:
                        rel_descriptions.append(f"{source} -> {target}: {desc}")
                if rel_descriptions:
                    contexts.append("[Relationships]:\n" + "\n".join(rel_descriptions[:15]))  # Limit
        
        # Extract from 'claims' or 'Claims' if available
        claims_key = context_keys_lower.get('claims')
        if claims_key:
            claims = context_data[claims_key]
            if isinstance(claims, pd.DataFrame) and 'description' in claims.columns:
                claim_descriptions = []
                for _, row in claims.iterrows():
                    desc = str(row.get('description', ''))
                    if desc and len(desc) > 10:
                        claim_descriptions.append(desc)
                if claim_descriptions:
                    contexts.append("[Claims]:\n" + "\n".join(claim_descriptions[:10]))
    
    elif isinstance(context_data, list):
        # Handle list of items
        for item in context_data:
            if isinstance(item, dict):
                # Recursively extract from nested dict
                contexts.extend(extract_context_texts_from_graphrag(item, max_text_length, logger))
            elif isinstance(item, pd.DataFrame):
                if 'text' in item.columns:
                    for _, row in item.iterrows():
                        text = str(row.get('text', ''))
                        if text:
                            contexts.append(text[:max_text_length])
            else:
                text = str(item)
                if text and len(text) > 10:
                    contexts.append(text[:max_text_length])
    
    elif isinstance(context_data, pd.DataFrame):
        # Handle single DataFrame
        if 'text' in context_data.columns:
            for _, row in context_data.iterrows():
                text = str(row.get('text', ''))
                if text:
                    contexts.append(text[:max_text_length])
        elif 'content' in context_data.columns:
            for _, row in context_data.iterrows():
                content = str(row.get('content', ''))
                if content:
                    contexts.append(content[:max_text_length])
    
    elif isinstance(context_data, str):
        if context_data and len(context_data) > 10:
            contexts.append(context_data)
    
    # IMPROVED: Handle empty context gracefully while still logging the issue
    if not contexts:
        # Determine if this is truly an error or just empty results
        is_truly_empty = (
            context_data is None or 
            (isinstance(context_data, dict) and len(context_data) == 0) or
            (isinstance(context_data, list) and len(context_data) == 0) or
            (isinstance(context_data, str) and len(context_data.strip()) == 0)
        )
        
        if is_truly_empty:
            # Log as warning, not error - empty context can be valid in some cases
            warning_msg = (
                f"No contexts extracted from GraphRAG response. "
                f"Response type: '{type(context_data).__name__}'. "
            )
            if isinstance(context_data, dict):
                available_keys = list(context_data.keys())
                warning_msg += f"Available keys: {available_keys}. "
            
            if logger:
                logger.warning(warning_msg + "RAGAS metrics may be affected.")
            
            # Return a placeholder context indicating no sources found
            # This allows metrics to still compute (with low scores) rather than failing
            contexts.append("[No relevant sources found in knowledge base for this query]")
        else:
            # Non-empty but unrecognized format - this is an actual error
            error_msg = (
                f"CRITICAL ERROR: Failed to extract contexts from GraphRAG response. "
                f"Response type: '{type(context_data).__name__}'. "
            )
            if isinstance(context_data, dict):
                available_keys = list(context_data.keys())
                error_msg += f"Available keys: {available_keys}. "
                error_msg += f"Expected keys: 'sources', 'Sources', 'reports', etc. "
            
            str_repr = str(context_data)
            error_msg += f"Response length: {len(str_repr)} chars. "
            
            if logger:
                logger.error(error_msg + "RAGAS metrics cannot be computed reliably without valid contexts.")
            
            # Return error message as context so the issue is traceable
            contexts.append(f"[ERROR: Context extraction failed - {error_msg}]")
    
    return contexts


class QueryRunner(ABC):
    """Abstract base class for query execution."""
    
    @abstractmethod
    async def run_query(self, question: str, context: Any = None, 
                       prompt_template: Optional[str] = None) -> Tuple[str, str, List[str]]:
        """Run query and return (raw_answer, processed_answer, contexts)."""
        pass


class GraphRAGRunner(QueryRunner):
    """Runner for GraphRAG-based queries (local_search)."""
    
    def __init__(self, config: Any, index_files: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.index_files = index_files
        self.logger = logger
    
    async def run_query(self, question: str, context: Any = None,
                       prompt_template: Optional[str] = None) -> Tuple[str, str, List[str]]:
        """Run GraphRAG local search query.
        
        OPTIMIZED: Runs local_search only ONCE per query instead of twice.
        When a prompt_template is provided, we use it directly instead of 
        running once without and once with the template.
        """
        if not GRAPHRAG_AVAILABLE:
            raise ImportError("GraphRAG is not available")
        
        # Extract question text if question is a dict
        if isinstance(question, dict):
            question_text = question.get('question', str(question))
        else:
            question_text = str(question)
        
        try:
            # Prepare config - use custom prompt if provided
            config_to_use = self.config
            temp_prompt_path = None
            
            if prompt_template:
                # Create a custom prompt content by replacing the GraphRAG template format
                custom_prompt = self._convert_evaluation_prompt_to_graphrag_format(prompt_template)
                
                # Create temporary prompt file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                    temp_file.write(custom_prompt)
                    temp_prompt_path = temp_file.name
                
                # Clone the config to avoid race conditions with concurrent requests
                config_to_use = deepcopy(self.config)
                
                # Modify the cloned config to use our custom prompt
                relative_path = os.path.relpath(temp_prompt_path, config_to_use.root_dir)
                config_to_use.local_search.prompt = relative_path
            
            try:
                # Run local search ONCE with the appropriate config
                raw_answer, context_data = await local_search(
                    config=config_to_use,
                    entities=self.index_files.get('entities'),
                    communities=self.index_files.get('communities'),
                    community_reports=self.index_files.get('community_reports'),
                    text_units=self.index_files.get('text_units'),
                    relationships=self.index_files.get('relationships'),
                    covariates=self.index_files.get('covariates'),
                    community_level=2,
                    response_type='Short Phrase',
                    query=question_text,
                )
                
                # Clean <think> tags from reasoning models (e.g., GPT-Oss)
                raw_answer = clean_think_tags(str(raw_answer))
                
                # For local_search, raw_answer and processed_answer are the same 
                # since we only run once with the appropriate prompt
                processed_answer = raw_answer
                
            finally:
                # Clean up temporary file if created
                if temp_prompt_path:
                    try:
                        os.unlink(temp_prompt_path)
                    except OSError as e:
                        self.logger.warning(f"Failed to delete temp prompt file {temp_prompt_path}: {e}")
            
            # Convert context to list of strings using proper extraction
            # This extracts full text content from GraphRAG context for RAGAS evaluation
            contexts = extract_context_texts_from_graphrag(context_data, logger=self.logger)
            
            return raw_answer, processed_answer, contexts
            
        except Exception as e:
            self.logger.error(f"GraphRAG query failed: {str(e)}")
            raise
    
    def _convert_evaluation_prompt_to_graphrag_format(self, evaluation_prompt: str) -> str:
        """Convert evaluation framework prompt template to GraphRAG format."""
        # Remove the {question} placeholder patterns since GraphRAG handles the question separately
        graphrag_prompt = evaluation_prompt.replace("Question: {question}\n\nAnswer:", "").strip()
        graphrag_prompt = graphrag_prompt.replace("Question: {question}\n\nReasoning:", "").strip()
        graphrag_prompt = graphrag_prompt.replace("Question: {question}", "").strip()
        graphrag_prompt = graphrag_prompt.replace("{question}", "the user's question").strip()
        
        # Add GraphRAG-specific formatting variables
        graphrag_formatted = f"""
---Role---

{graphrag_prompt}

---Goal---

Generate a response that answers the user's question directly and concisely, following the instructions above.

---Target response length and format---

{{response_type}}

---Data tables---

{{context_data}}

---Goal---

Generate a response that answers the user's question directly and concisely, following the instructions above.
"""
        
        return graphrag_formatted


class GlobalSearchRunner(QueryRunner):
    """Runner for GraphRAG global search queries."""
    
    def __init__(self, config: Any, index_files: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.index_files = index_files
        self.logger = logger
    
    async def run_query(self, question: str, context: Any = None, 
                       prompt_template: Optional[str] = None) -> Tuple[str, str, List[str]]:
        """Run GraphRAG global search query.
        
        OPTIMIZED: Runs global_search only ONCE per query instead of twice.
        When a prompt_template is provided, we use it directly instead of 
        running once without and once with the template.
        """
        if not GRAPHRAG_AVAILABLE:
            raise ImportError("GraphRAG is not available")
        
        # Extract question text if question is a dict
        if isinstance(question, dict):
            question_text = question.get('question', str(question))
        else:
            question_text = str(question)
        
        try:
            # Prepare config - use custom prompts if provided
            config_to_use = self.config
            temp_map_path = None
            temp_reduce_path = None
            
            if prompt_template:
                # Create custom prompts in GraphRAG format
                custom_map_prompt = self._convert_evaluation_prompt_to_global_map_format(prompt_template)
                custom_reduce_prompt = self._convert_evaluation_prompt_to_global_reduce_format(prompt_template)
                
                # Create temporary prompt files
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_map_file:
                    temp_map_file.write(custom_map_prompt)
                    temp_map_path = temp_map_file.name
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_reduce_file:
                    temp_reduce_file.write(custom_reduce_prompt)
                    temp_reduce_path = temp_reduce_file.name
                
                # Clone the config to avoid race conditions with concurrent requests
                config_to_use = deepcopy(self.config)
                
                # Modify the cloned config to use our custom prompts
                config_to_use.global_search.map_prompt = os.path.relpath(temp_map_path, config_to_use.root_dir)
                config_to_use.global_search.reduce_prompt = os.path.relpath(temp_reduce_path, config_to_use.root_dir)
            
            try:
                # Run global search ONCE with the appropriate config
                raw_answer, context_data = await global_search(
                    config=config_to_use,
                    entities=self.index_files.get('entities'),
                    communities=self.index_files.get('communities'),
                    community_reports=self.index_files.get('community_reports'),
                    community_level=2,
                    dynamic_community_selection=False,
                    response_type='Short Phrase',
                    query=question_text,
                )
                
                # Clean <think> tags from reasoning models (e.g., GPT-Oss)
                raw_answer = clean_think_tags(str(raw_answer))
                
                # For global_search, raw_answer and processed_answer are the same 
                # since we only run once with the appropriate prompts
                processed_answer = raw_answer
                
            finally:
                # Clean up temporary files if created
                if temp_map_path:
                    try:
                        os.unlink(temp_map_path)
                    except OSError as e:
                        self.logger.warning(f"Failed to delete temp map prompt file {temp_map_path}: {e}")
                if temp_reduce_path:
                    try:
                        os.unlink(temp_reduce_path)
                    except OSError as e:
                        self.logger.warning(f"Failed to delete temp reduce prompt file {temp_reduce_path}: {e}")
            
            # Convert context to list of strings using proper extraction
            # This extracts full text content from GraphRAG context for RAGAS evaluation
            contexts = extract_context_texts_from_graphrag(context_data, logger=self.logger)
            
            return raw_answer, processed_answer, contexts
            
        except Exception as e:
            self.logger.error(f"GraphRAG global search failed: {str(e)}")
            raise
    
    def _convert_evaluation_prompt_to_global_map_format(self, evaluation_prompt: str) -> str:
        """Convert evaluation framework prompt template to GraphRAG global search map format."""
        # Remove the {question} placeholder patterns since GraphRAG handles the question separately
        core_instructions = evaluation_prompt.replace("Question: {question}\n\nAnswer:", "").strip()
        core_instructions = core_instructions.replace("Question: {question}\n\nReasoning:", "").strip()
        core_instructions = core_instructions.replace("Question: {question}", "").strip()
        core_instructions = core_instructions.replace("{question}", "the user's question").strip()
        
        # Remove any trailing "Answer:" lines that might be left
        core_instructions = core_instructions.replace("Answer:", "").strip()
        
        # CRITICAL: Escape any remaining curly braces in core_instructions to prevent
        # Python's str.format() from interpreting them as placeholders when GraphRAG
        # later calls .format() on this prompt template
        core_instructions = core_instructions.replace("{", "{{").replace("}", "}}")
        
        # Create global search map prompt format
        global_map_prompt = f"""
---Role---

{core_instructions}

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, following the instructions above and summarizing all relevant information in the input data tables.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point following the above instructions.
- Importance Score: An integer score between 0-100 that indicates how important the point is.

The response should be JSON formatted as follows:
{{{{
    "points": [
        {{{{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}}}},
        {{{{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}}}
    ]
}}}}

Points supported by data should list the relevant reports as references.
Do not list more than 5 record ids in a single reference.

Limit your response length to {{max_length}} words.

---Data tables---

{{context_data}}
"""
        return global_map_prompt
    
    def _convert_evaluation_prompt_to_global_reduce_format(self, evaluation_prompt: str) -> str:
        """Convert evaluation framework prompt template to GraphRAG global search reduce format."""
        # Remove the {question} placeholder patterns since GraphRAG handles the question separately
        core_instructions = evaluation_prompt.replace("Question: {question}\n\nAnswer:", "").strip()
        core_instructions = core_instructions.replace("Question: {question}\n\nReasoning:", "").strip()
        core_instructions = core_instructions.replace("Question: {question}", "").strip()
        core_instructions = core_instructions.replace("{question}", "the user's question").strip()
        
        # CRITICAL: Escape any remaining curly braces in core_instructions to prevent
        # Python's str.format() from interpreting them as placeholders when GraphRAG
        # later calls .format() on this prompt template
        core_instructions = core_instructions.replace("{", "{{").replace("}", "}}")
        
        # Create global search reduce prompt format
        global_reduce_prompt = f"""
---Role---

{core_instructions}

---Goal---

Generate a final response that answers the user's question following the instructions above, based on the analyst reports provided below.

Synthesize the key points into a coherent response that directly answers the question following all the specified instructions.

---Target response length and format---

{{response_type}}

---Analyst Reports---

{{report_data}}

---Goal---

Generate a final response that answers the user's question following the instructions above.
"""
        return global_reduce_prompt


class RAGRunner(QueryRunner):
    """Runner for RAG-based queries (basic_search)."""
    
    def __init__(self, config: Any, index_files: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.index_files = index_files
        self.logger = logger
    
    async def run_query(self, question: str, context: Any = None, 
                       prompt_template: Optional[str] = None) -> Tuple[str, str, List[str]]:
        """Run RAG basic search query.
        
        OPTIMIZED: Runs basic_search only ONCE per query instead of twice.
        When a prompt_template is provided, we use it directly instead of 
        running once without and once with the template.
        """
        if not GRAPHRAG_AVAILABLE:
            raise ImportError("GraphRAG is not available")
        
        # Extract question text if question is a dict
        if isinstance(question, dict):
            question_text = question.get('question', str(question))
        else:
            question_text = str(question)
        
        try:
            # Prepare config - use custom prompt if provided
            config_to_use = self.config
            temp_prompt_path = None
            
            if prompt_template:
                # Create a custom prompt content in basic search format
                custom_prompt = self._convert_evaluation_prompt_to_basic_search_format(prompt_template)
                
                # Create temporary prompt file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                    temp_file.write(custom_prompt)
                    temp_prompt_path = temp_file.name
                
                # Clone the config to avoid race conditions with concurrent requests
                config_to_use = deepcopy(self.config)
                
                # Modify the cloned config to use our custom prompt
                relative_path = os.path.relpath(temp_prompt_path, config_to_use.root_dir)
                config_to_use.basic_search.prompt = relative_path
            
            try:
                # Run basic search ONCE with the appropriate config
                raw_answer, context_data = await basic_search(
                    config=config_to_use,
                    text_units=self.index_files.get('text_units'),
                    query=question_text,
                )
                
                # Clean <think> tags from reasoning models (e.g., GPT-Oss)
                raw_answer = clean_think_tags(str(raw_answer))
                
                # For basic_search, raw_answer and processed_answer are the same 
                # since we only run once with the appropriate prompt
                processed_answer = raw_answer
                
            finally:
                # Clean up temporary file if created
                if temp_prompt_path:
                    try:
                        os.unlink(temp_prompt_path)
                    except OSError as e:
                        self.logger.warning(f"Failed to delete temp prompt file {temp_prompt_path}: {e}")
            
            # Convert context to list of strings using proper extraction
            # This extracts full text content from GraphRAG context for RAGAS evaluation
            contexts = extract_context_texts_from_graphrag(context_data, logger=self.logger)
            
            return raw_answer, processed_answer, contexts
            
        except Exception as e:
            self.logger.error(f"RAG query failed: {str(e)}")
            raise
    
    def _convert_evaluation_prompt_to_basic_search_format(self, evaluation_prompt: str) -> str:
        """Convert evaluation framework prompt template to GraphRAG basic search format."""
        # Remove the {question} placeholder patterns since GraphRAG handles the question separately
        core_instructions = evaluation_prompt.replace("Question: {question}\n\nAnswer:", "").strip()
        core_instructions = core_instructions.replace("Question: {question}\n\nReasoning:", "").strip()
        core_instructions = core_instructions.replace("Question: {question}", "").strip()
        core_instructions = core_instructions.replace("{question}", "the user's question").strip()
        
        # Add GraphRAG basic search format
        basic_search_prompt = f"""
---Role---

{core_instructions}

---Goal---

Generate a response following the instructions above that responds to the user's question, summarizing all relevant information in the input data tables.

Use the data provided in the data tables below as the primary context for generating the response.

If you don't know the answer or if the input data tables do not contain sufficient information, just say so. Do not make anything up.

Points supported by data should list their data references as follows:
"This is an example sentence supported by multiple data references [Data: Sources (record ids)]."

Do not list more than 5 record ids in a single reference.

---Target response length and format---

{{response_type}}

---Data tables---

{{context_data}}

---Goal---

Generate a response following the instructions above that responds to the user's question.
"""
        return basic_search_prompt


class DirectLLMRunner(QueryRunner):
    """Runner for direct LLM queries with evidence context.
    
    Uses chunk-based processing for large documents to avoid hallucination issues
    that occur when context exceeds model-specific limits.
    
    VERIFIED AIME API CONTEXT LIMITS (December 2025):
    - llama4_chat:   100,000 chars (~25K tokens)
    - mistral_chat:  250,000 chars (~62K tokens)
    - llama3_chat:   350,000 chars (~87K tokens)
    - qwen3_chat:    350,000 chars (~87K tokens)
    - gpt_oss_chat:  350,000 chars (~87K tokens)
    
    AGGREGATION STRATEGIES:
    - "tree_summarize": Hierarchical tree-based aggregation (LlamaIndex style)
      Faster than refine, good quality. Processes chunks in parallel, then
      combines answers hierarchically until single answer remains.
    - "refine": Sequential iterative refinement (LangChain style)
      Slower but potentially better for complex reasoning. Each chunk
      refines the previous answer.
    - "map_reduce": Process all chunks, then single aggregation call.
      Fastest but may lose nuance with many chunks.
    
    PERFORMANCE OPTIMIZATIONS:
    - Model-specific context limits to minimize unnecessary chunking
    - Parallel chunk processing (up to 4 chunks simultaneously)
    - Reduced chunk overlap (1000 chars) for context continuity
    - Large chunk size (80K) to reduce total chunks needed
    """
    
    # Model-specific context limits (verified via testing)
    # Using 85% of max to leave room for prompt overhead
    MODEL_CONTEXT_LIMITS = {
        'llama4_chat': 90000,      # 100K limit, use 90K safe
        'mistral_chat': 225000,    # 250K limit, use 225K safe
        'llama3_chat': 230000,     # 65K tokens (~262K chars), use 230K safe
        'qwen3_chat': 230000,      # Conservative - verify actual limit
        'gpt_oss_chat': 230000,    # Conservative - verify actual limit
    }
    DEFAULT_CONTEXT_LIMIT = 90000  # Conservative default (llama4 limit)
    
    # Chunk size as percentage of context limit (use 85% to leave room for question/prompt)
    CHUNK_SIZE_RATIO = 0.85
    CHUNK_OVERLAP = 500    # Overlap for context continuity
    
    # Parallel processing - increase for better throughput
    MAX_PARALLEL_CHUNKS = 8  # Process up to 8 chunks in parallel (semaphore-controlled)
    
    # Aggregation strategy: "tree_summarize", "refine", or "map_reduce"
    AGGREGATION_STRATEGY = "map_reduce"  # Default: fastest with good quality for factual Q&A
    
    # Request timeout and retry settings for faster failure detection
    CHUNK_TIMEOUT = 240  # Seconds before timing out a chunk request
    MAX_RETRIES = 2      # Retry failed chunks (reduced from adapter's default)
    
    # Instance-level cache tracking (FIX: CRITICAL-001 - moved from class-level to instance-level)
    # Each instance now has its own cache with async lock for thread safety
    
    def __init__(self, llm_adapter: Any, logger: logging.Logger, model_name: str = None,
                 aggregation_strategy: str = None):
        self.llm_adapter = llm_adapter
        self.logger = logger
        self.model_name = model_name
        self.aggregation_strategy = aggregation_strategy or self.AGGREGATION_STRATEGY
        
        # FIX: CRITICAL-001 - Instance-level cache with async lock for thread safety
        self._chunk_cache: Dict[str, str] = {}
        self._cache_lock = asyncio.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Set context limit based on model
        self.safe_context_size = self.MODEL_CONTEXT_LIMITS.get(
            model_name, self.DEFAULT_CONTEXT_LIMIT
        )
        
        # Calculate chunk size based on model's context limit (85% to leave room for prompt)
        # This reduces total chunks needed for models with larger context windows
        self.max_chunk_size = int(self.safe_context_size * self.CHUNK_SIZE_RATIO)
        
        self.logger.info(
            f"DirectLLMRunner initialized for model '{model_name}' "
            f"with context limit: {self.safe_context_size:,} chars, "
            f"chunk size: {self.max_chunk_size:,} chars, "
            f"aggregation strategy: {self.aggregation_strategy}"
        )
    
    @classmethod
    def _get_cache_key(cls, question: str, chunk_hash: str) -> str:
        """Generate cache key from question and chunk content hash."""
        import hashlib
        q_hash = hashlib.md5(question.encode()).hexdigest()[:8]
        return f"{q_hash}:{chunk_hash}"
    
    @classmethod
    def _get_chunk_hash(cls, chunk: str) -> str:
        """Generate hash of chunk content for caching."""
        import hashlib
        return hashlib.md5(chunk.encode()).hexdigest()[:16]
    
    def clear_cache(self):
        """Clear the chunk cache (instance-level, thread-safe)."""
        # Note: This is now instance-level, no lock needed for clear
        self._chunk_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics (instance-level)."""
        return {
            "cache_size": len(self._chunk_cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        }
    
    def _split_context_into_chunks(self, context: str) -> List[str]:
        """Split large context into overlapping chunks for processing.
        
        This prevents hallucination issues that occur with very large contexts
        by processing the document in manageable pieces.
        
        Chunk size is determined by model's context limit (set in __init__).
        """
        if len(context) <= self.safe_context_size:
            return [context]
        
        chunks = []
        start = 0
        
        while start < len(context):
            end = start + self.max_chunk_size
            
            # Try to break at a paragraph or sentence boundary
            if end < len(context):
                # Look for paragraph break first
                para_break = context.rfind('\n\n', start + self.max_chunk_size // 2, end)
                if para_break > start:
                    end = para_break
                else:
                    # Look for sentence break
                    for sep in ['. ', '.\n', '? ', '!\n']:
                        sent_break = context.rfind(sep, start + self.max_chunk_size // 2, end)
                        if sent_break > start:
                            end = sent_break + len(sep)
                            break
            
            chunk = context[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start with overlap
            start = end - self.CHUNK_OVERLAP if end < len(context) else end
        
        self.logger.info(
            f"Split context ({len(context):,} chars) into {len(chunks)} chunks "
            f"(~{self.max_chunk_size:,} chars each with {self.CHUNK_OVERLAP:,} overlap)"
        )
        
        return chunks
    
    async def _process_single_chunk(self, question: str, chunk: str, chunk_idx: int, 
                                   total_chunks: int, prompt_template: Optional[str] = None) -> str:
        """Process a single chunk and return the answer.
        
        Uses caching to avoid reprocessing identical question+chunk pairs.
        FIX: CRITICAL-001 - Uses async lock for thread-safe cache access.
        """
        # Check cache first (thread-safe)
        chunk_hash = self._get_chunk_hash(chunk)
        cache_key = self._get_cache_key(question, chunk_hash)
        
        # FIX: CRITICAL-001 - Use async lock for thread-safe cache access
        async with self._cache_lock:
            if cache_key in self._chunk_cache:
                self._cache_hits += 1
                self.logger.debug(f"Chunk {chunk_idx+1} cache HIT")
                return self._chunk_cache[cache_key]
            self._cache_misses += 1
        
        # Add chunk context to help model understand it's seeing partial content
        chunk_header = ""
        if total_chunks > 1:
            chunk_header = f"[NOTE: This is part {chunk_idx + 1} of {total_chunks} of a larger document. Focus on information relevant to the question.]\n\n"
        
        contextualized_chunk = chunk_header + chunk
        
        if prompt_template:
            formatted_template = prompt_template.replace("{question}", question)
            
            if "{question}" in prompt_template:
                full_query = f"{formatted_template}\n\n[CONTEXT]: {contextualized_chunk}"
            else:
                full_query = f"{formatted_template}[CONTEXT]: {contextualized_chunk}\n[QUESTION]: {question}"
        else:
            full_query = f"[CONTEXT]: {contextualized_chunk}\n[QUESTION]: {question}"
        
        response = await self.llm_adapter.ainvoke(full_query)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Clean <think> tags from reasoning models (e.g., GPT-Oss)
        answer = clean_think_tags(answer)
        
        # Cache the result (only cache successful responses, thread-safe)
        if not answer.startswith('[ERROR'):
            async with self._cache_lock:
                self._chunk_cache[cache_key] = answer
        
        return answer
    
    def _filter_valid_answers(self, chunk_answers: List[str]) -> List[str]:
        """Filter out empty, error, or non-informative answers."""
        valid_answers = []
        for a in chunk_answers:
            if not a:
                continue
            if a.startswith('[ERROR'):
                continue
            # Skip non-informative answers
            lower_a = a.lower()
            if any(phrase in lower_a for phrase in [
                "i don't know", "i do not know", "cannot find", "not found",
                "no information", "not mentioned", "unable to find",
                "the context does not", "the document does not"
            ]):
                continue
            valid_answers.append(a)
        return valid_answers
    
    async def _combine_two_answers(self, question: str, answer1: str, answer2: str) -> str:
        """Combine two answers into one using LLM."""
        combine_prompt = f"""You are combining two partial answers to a question into a single comprehensive answer.

QUESTION: {question}

ANSWER 1:
{answer1}

ANSWER 2:
{answer2}

INSTRUCTIONS:
1. Merge the information from both answers into a single, coherent response.
2. If the answers agree, combine them comprehensively.
3. If they conflict, prefer the more specific or complete information.
4. Do not mention that you are combining answers - just provide the final answer.
5. Be concise but complete.

COMBINED ANSWER:"""
        
        response = await self.llm_adapter.ainvoke(combine_prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    async def _tree_summarize(self, question: str, answers: List[str]) -> str:
        """Hierarchical tree-based aggregation (LlamaIndex style).
        
        Combines answers in pairs recursively until a single answer remains.
        This is faster than refine (parallel at each level) while maintaining quality.
        
        Example with 5 answers:
        Level 0: [A1, A2, A3, A4, A5]
        Level 1: [combine(A1,A2), combine(A3,A4), A5] -> [B1, B2, B3]
        Level 2: [combine(B1,B2), B3] -> [C1, C2]
        Level 3: [combine(C1,C2)] -> [D1] -> Final answer
        """
        if len(answers) == 0:
            return "[No relevant information found in the document]"
        if len(answers) == 1:
            return answers[0]
        
        current_level = answers
        level = 0
        
        while len(current_level) > 1:
            level += 1
            next_level = []
            
            self.logger.info(f"Tree summarize level {level}: combining {len(current_level)} answers...")
            
            # Pair up answers for combining
            pairs = []
            for i in range(0, len(current_level) - 1, 2):
                pairs.append((current_level[i], current_level[i + 1]))
            
            # If odd number, carry the last one forward
            has_odd = len(current_level) % 2 == 1
            
            # Process pairs in parallel
            async def combine_pair(idx: int, pair: Tuple[str, str]) -> Tuple[int, str]:
                try:
                    combined = await self._combine_two_answers(question, pair[0], pair[1])
                    return (idx, combined)
                except Exception as e:
                    self.logger.warning(f"Pair combination failed: {str(e)}")
                    # Fallback: return longer answer
                    return (idx, pair[0] if len(pair[0]) >= len(pair[1]) else pair[1])
            
            # Process pairs in parallel batches
            combined_results = [None] * len(pairs)
            for batch_start in range(0, len(pairs), self.MAX_PARALLEL_CHUNKS):
                batch_end = min(batch_start + self.MAX_PARALLEL_CHUNKS, len(pairs))
                batch_pairs = [(i, pairs[i]) for i in range(batch_start, batch_end)]
                
                tasks = [combine_pair(idx, pair) for idx, pair in batch_pairs]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.warning(f"Batch combine failed: {str(result)}")
                    else:
                        idx, combined = result
                        combined_results[idx] = combined
            
            # Build next level
            next_level = [r for r in combined_results if r is not None]
            
            # Add the odd one if exists
            if has_odd:
                next_level.append(current_level[-1])
            
            self.logger.info(f"Tree summarize level {level} complete: {len(current_level)} -> {len(next_level)} answers")
            current_level = next_level
        
        return current_level[0] if current_level else "[Aggregation failed]"
    
    async def _refine_answers(self, question: str, answers: List[str]) -> str:
        """Sequential iterative refinement (LangChain style).
        
        Processes answers one by one, refining the accumulated answer each time.
        Slower but potentially better for complex reasoning.
        """
        if len(answers) == 0:
            return "[No relevant information found in the document]"
        if len(answers) == 1:
            return answers[0]
        
        current_answer = answers[0]
        
        for i, new_answer in enumerate(answers[1:], start=2):
            self.logger.info(f"Refine step {i}/{len(answers)}: refining with new information...")
            
            refine_prompt = f"""You have an existing answer to a question, and new information from another part of the document.
Refine the existing answer by incorporating the new information.

QUESTION: {question}

EXISTING ANSWER:
{current_answer}

NEW INFORMATION:
{new_answer}

INSTRUCTIONS:
1. If the new information adds to the existing answer, incorporate it.
2. If the new information contradicts, prefer the more specific/complete information.
3. If the new information is irrelevant or says "I don't know", keep the existing answer.
4. Provide a refined, comprehensive answer.

REFINED ANSWER:"""
            
            try:
                response = await self.llm_adapter.ainvoke(refine_prompt)
                current_answer = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                self.logger.warning(f"Refine step {i} failed: {str(e)}, keeping previous answer")
        
        return current_answer
    
    async def _map_reduce_answers(self, question: str, answers: List[str]) -> str:
        """Simple map-reduce: single aggregation call for all answers."""
        answers_text = "\n\n---\n\n".join([
            f"[Answer from Document Part {i+1}]:\n{answer}" 
            for i, answer in enumerate(answers)
        ])
        
        aggregation_prompt = f"""You have been given multiple partial answers to a question, each derived from different parts of a large document.
Your task is to synthesize these into a single, coherent, and accurate answer.

IMPORTANT INSTRUCTIONS:
1. If the partial answers agree, combine the information into a comprehensive response.
2. If the partial answers conflict, prefer information that appears more complete or specific.
3. If some parts say "I don't know" or lack information, ignore those and use the informative parts.
4. Keep your final answer concise and directly address the question.
5. Do not mention that you are synthesizing from multiple parts - just give the final answer.

QUESTION: {question}

PARTIAL ANSWERS FROM DOCUMENT PARTS:
{answers_text}

SYNTHESIZED FINAL ANSWER:"""
        
        response = await self.llm_adapter.ainvoke(aggregation_prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    async def _aggregate_chunk_answers(self, question: str, chunk_answers: List[str], 
                                       prompt_template: Optional[str] = None) -> str:
        """Aggregate answers from multiple chunks into a single coherent answer.
        
        Uses the configured aggregation strategy:
        - tree_summarize: Hierarchical combination (default, good balance)
        - refine: Sequential refinement (slower, potentially more accurate)
        - map_reduce: Single aggregation (fastest, may lose nuance)
        """
        # Filter to valid answers
        valid_answers = self._filter_valid_answers(chunk_answers)
        
        if not valid_answers:
            return "[No relevant information found in the document]"
        
        if len(valid_answers) == 1:
            return valid_answers[0]
        
        # Apply configured aggregation strategy
        # NOTE: Removed the "longest answer" shortcut for fair evaluation.
        # All valid answers are now properly aggregated using the configured strategy.
        self.logger.info(f"Aggregating {len(valid_answers)} answers using '{self.aggregation_strategy}' strategy")
        
        if self.aggregation_strategy == "tree_summarize":
            return await self._tree_summarize(question, valid_answers)
        elif self.aggregation_strategy == "refine":
            return await self._refine_answers(question, valid_answers)
        else:  # map_reduce (default fallback)
            return await self._map_reduce_answers(question, valid_answers)
    
    async def run_query(self, question: str, context: Any = None, 
                       prompt_template: Optional[str] = None) -> Tuple[str, str, List[str]]:
        """Run direct LLM query with evidence context.
        
        For large contexts (>50K chars), uses chunk-based processing to avoid
        hallucination issues that occur with very long inputs.
        """
        try:
            # Prepare context
            if context is None:
                context = ""
            elif isinstance(context, list):
                context = "\n".join(context)
            else:
                context = str(context)
            
            context_size = len(context)
            use_chunking = context_size > self.safe_context_size
            
            if use_chunking:
                self.logger.info(
                    f"Large context detected ({context_size:,} chars > {self.safe_context_size:,}). "
                    f"Using chunk-based processing to avoid hallucination."
                )
            
            # Split context into chunks if needed
            chunks = self._split_context_into_chunks(context) if context else [""]
            
            # Process chunks
            if len(chunks) == 1:
                # Single chunk - original behavior
                chunk = chunks[0]
                
                # First get raw answer without prompt template
                if chunk:
                    raw_query = f"[CONTEXT]: {chunk}\n[QUESTION]: {question}"
                else:
                    raw_query = question
                
                raw_response = await self.llm_adapter.ainvoke(raw_query)
                raw_answer = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
                
                # Clean <think> tags from reasoning models (e.g., GPT-Oss)
                raw_answer = clean_think_tags(raw_answer)
                
                # Get processed answer with prompt template if provided
                processed_answer = raw_answer  # Default to raw answer
                if prompt_template:
                    # Replace the {question} placeholder in the template
                    formatted_template = prompt_template.replace("{question}", question)
                    
                    # If the template already contains the question, use it as-is
                    # Otherwise, append context and question sections
                    if "{question}" in prompt_template:
                        # Template format: includes question placeholder, so just add context if available
                        if chunk:
                            full_query = f"{formatted_template}\n\n[CONTEXT]: {chunk}"
                        else:
                            full_query = formatted_template
                    else:
                        # Old format: append context and question
                        if chunk:
                            full_query = f"{formatted_template}[CONTEXT]: {chunk}\n[QUESTION]: {question}"
                        else:
                            full_query = f"{formatted_template}[CONTEXT]: [MISSING]\n[QUESTION]: {question}"
                    
                    # Get response from LLM
                    response = await self.llm_adapter.ainvoke(full_query)
                    processed_answer = response.content if hasattr(response, 'content') else str(response)
                    
                    # Clean <think> tags from reasoning models (e.g., GPT-Oss)
                    processed_answer = clean_think_tags(processed_answer)
            else:
                # Multiple chunks - process ALL in parallel with semaphore (no batch waiting)
                self.logger.info(f"Processing {len(chunks)} chunks for question: {question[:100]}...")
                
                # Use semaphore to limit concurrent requests while keeping pipeline full
                semaphore = asyncio.Semaphore(self.MAX_PARALLEL_CHUNKS)
                
                async def process_chunk_with_semaphore(idx: int, chunk: str) -> Tuple[int, str]:
                    """Process a chunk with semaphore-controlled concurrency."""
                    async with semaphore:
                        try:
                            # Add timeout for faster failure detection
                            answer = await asyncio.wait_for(
                                self._process_single_chunk(
                                    question, chunk, idx, len(chunks), prompt_template
                                ),
                                timeout=self.CHUNK_TIMEOUT
                            )
                            self.logger.info(f"Chunk {idx+1}/{len(chunks)} done: {len(answer)} chars")
                            return (idx, answer)
                        except asyncio.TimeoutError:
                            self.logger.warning(f"Chunk {idx+1}/{len(chunks)} timed out after {self.CHUNK_TIMEOUT}s")
                            return (idx, "[ERROR: Request timed out]")
                        except Exception as e:
                            self.logger.warning(f"Chunk {idx+1}/{len(chunks)} failed: {str(e)}")
                            return (idx, f"[ERROR: {str(e)}]")
                
                # Launch ALL chunk tasks immediately - semaphore controls actual concurrency
                tasks = [process_chunk_with_semaphore(i, chunk) for i, chunk in enumerate(chunks)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results maintaining order
                chunk_answers = []
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.warning(f"Chunk task failed: {str(result)}")
                        chunk_answers.append("[ERROR: Processing failed]")
                    else:
                        idx, answer = result
                        chunk_answers.append(answer)
                
                # Aggregate answers
                raw_answer = " | ".join([f"[Chunk {i+1}]: {a[:200]}..." for i, a in enumerate(chunk_answers)])
                processed_answer = await self._aggregate_chunk_answers(question, chunk_answers, prompt_template)
                
                self.logger.info(
                    f"Aggregated {len(chunk_answers)} chunk answers into final answer "
                    f"({len(processed_answer)} chars)"
                )
            
            # Return raw_answer, processed_answer and context as list
            context_list = [context] if context else []
            return raw_answer, processed_answer, context_list
            
        except Exception as e:
            self.logger.error(f"Direct LLM query failed: {str(e)}")
            raise


class QueryRunnerFactory:
    """Factory for creating appropriate query runners."""
    
    @staticmethod
    def create_runner(method: str, config: Any, logger: logging.Logger, 
                     index_files: Optional[Dict[str, Any]] = None, llm_adapter: Any = None,
                     model_api: Any = None, aggregation_strategy: str = None) -> QueryRunner:
        """Create query runner based on method type.
        
        Args:
            method: Query method ('local_search', 'global_search', 'basic_search', 'llm_with_context')
            config: Configuration object
            logger: Logger instance
            index_files: Index files for GraphRAG/RAG methods
            llm_adapter: LLM adapter for direct LLM queries
            model_api: Model API instance (optional)
            aggregation_strategy: For llm_with_context, strategy for combining chunk answers:
                - "tree_summarize": Hierarchical combination (default, good balance)
                - "refine": Sequential refinement (slower, potentially more accurate)
                - "map_reduce": Single aggregation (fastest, may lose nuance)
        """
        
        if method == 'local_search':
            if not index_files:
                raise ValueError("Index files required for local_search")
            return GraphRAGRunner(config, index_files, logger)
            
        elif method == 'global_search':
            if not index_files:
                raise ValueError("Index files required for global_search")
            return GlobalSearchRunner(config, index_files, logger)
            
        elif method == 'basic_search':
            if not index_files:
                raise ValueError("Index files required for basic_search")
            return RAGRunner(config, index_files, logger)
            
        elif method == 'llm_with_context':
            if not llm_adapter:
                raise ValueError("LLM adapter required for llm_with_context")
            # Extract model name from llm_adapter or config for model-specific context limits
            model_name = None
            # Try multiple sources for model name:
            # 1. AIMEAPIAdapter's model_api.endpoint_name (direct AIME API)
            if hasattr(llm_adapter, 'model_api'):
                model_name = getattr(llm_adapter.model_api, 'endpoint_name', None)
                # Also check model_name attribute (for AimeAPIProvider wrapped in model_api)
                if not model_name:
                    model_name = getattr(llm_adapter.model_api, 'model_name', None)
            # 2. LangChain adapter wrapping AimeAPIProvider
            if not model_name and hasattr(llm_adapter, 'llm'):
                model_name = getattr(llm_adapter.llm, 'model_name', None)
                if not model_name:
                    model_name = getattr(llm_adapter.llm, 'endpoint_name', None)
            # 3. Direct model_name attribute on adapter
            if not model_name:
                model_name = getattr(llm_adapter, 'model_name', None)
            # 4. Fallback: try config.models.default_chat_model.model
            if not model_name and config and hasattr(config, 'models'):
                default_chat = getattr(config.models, 'default_chat_model', None)
                if default_chat:
                    model_name = getattr(default_chat, 'model', None)
            # 5. Final fallback: try config.local_search.chat_model_id (GraphRAG config)
            if not model_name and config and hasattr(config, 'local_search'):
                chat_model_id = getattr(config.local_search, 'chat_model_id', None)
                if chat_model_id:
                    model_config = config.get_language_model_config(chat_model_id) if hasattr(config, 'get_language_model_config') else None
                    if model_config:
                        model_name = getattr(model_config, 'model', None)
            
            logger.info(f"Detected model name for DirectLLMRunner: {model_name}")
            return DirectLLMRunner(
                llm_adapter, logger, 
                model_name=model_name,
                aggregation_strategy=aggregation_strategy
            )
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def get_method_architecture(method: str) -> str:
        """Get the architecture type for a given method."""
        if method in ['local_search', 'global_search']:
            return 'graphrag'
        elif method == 'basic_search':
            return 'rag'
        elif method == 'llm_with_context':
            return 'direct_llm'
        else:
            return 'unknown' 