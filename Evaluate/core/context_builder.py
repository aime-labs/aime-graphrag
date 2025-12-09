"""
Context Builder for llm_with_context method.

Provides a shared utility for building context from novel/medical JSON files,
ensuring consistency between unified (evaluator) mode and benchmark mode.
"""

import logging
from typing import Dict, Optional


def build_llm_context(
    novel_contexts: Dict[str, str],
    primary_source: str,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Build context for llm_with_context method by including ALL documents.
    
    This ensures the LLM has access to the same corpus as GraphRAG methods,
    providing a fair comparison between direct LLM and GraphRAG approaches.
    
    Args:
        novel_contexts: Dictionary mapping source IDs to document content
                       (loaded from novel.json or medical.json)
        primary_source: The source identifier for the question's originating document
                       (used to mark which document the question comes from)
        logger: Optional logger for debugging information
    
    Returns:
        Combined context string with all documents, with the primary document marked.
        Returns empty string if no contexts are available.
    
    Example output:
        === DOCUMENT (Source: doc1) [PRIMARY - Question Source] ===
        <content of doc1>
        
        === DOCUMENT (Source: doc2) ===
        <content of doc2>
    """
    if not novel_contexts:
        if logger:
            logger.debug("build_llm_context: No contexts available")
        return ""
    
    # Include ALL documents from novel.json/medical.json
    contexts = []
    total_size = 0
    
    for doc_source, doc_context in novel_contexts.items():
        if doc_context:
            # Mark the primary document (where question originates) for reference
            if str(doc_source) == str(primary_source):
                contexts.append(f"=== DOCUMENT (Source: {doc_source}) [PRIMARY - Question Source] ===\n{doc_context}")
            else:
                contexts.append(f"=== DOCUMENT (Source: {doc_source}) ===\n{doc_context}")
            total_size += len(doc_context)
    
    combined_context = "\n\n".join(contexts)
    
    if logger:
        logger.info(
            f"llm_with_context: question_source={primary_source}, ALL {len(contexts)} documents included, "
            f"total {total_size:,} chars (chunking will be applied if >50K chars)"
        )
    
    return combined_context
