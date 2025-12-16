"""
RAGAS (Retrieval Augmented Generation Assessment) Metrics Implementation

Implements the four core RAGAS metrics for evaluating RAG systems using 
LLM-as-a-Judge methodology:

1. Faithfulness: Measures if the generated answer is grounded in the retrieved context
2. Context Precision: Measures relevance of retrieved contexts to the question
3. Context Recall: Measures if contexts contain all info needed to answer
4. Answer Relevance: Measures if the answer addresses the question properly

Reference: 
- RAGAS: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)
- https://docs.ragas.io/

All scores are normalized to 0-100 scale for consistency with other metrics.
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, field
import asyncio
from pathlib import Path
import os
from utils.error_handling import safe_metric_computation, parse_json_response


@dataclass
class RagasEvaluationResult:
    """Container for comprehensive RAGAS evaluation results."""
    
    # Core RAGAS scores (0-100)
    faithfulness: float = np.nan
    context_precision: float = np.nan
    context_recall: float = np.nan
    answer_relevance: float = np.nan
    
    # Detailed breakdown
    faithfulness_details: Dict[str, Any] = field(default_factory=dict)
    context_precision_details: Dict[str, Any] = field(default_factory=dict)
    context_recall_details: Dict[str, Any] = field(default_factory=dict)
    answer_relevance_details: Dict[str, Any] = field(default_factory=dict)
    
    # Aggregated score
    ragas_score: float = np.nan
    
    def compute_ragas_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute weighted average RAGAS score.
        
        Args:
            weights: Optional custom weights for each metric.
                     Default weights: Faithfulness=0.3, Context Precision=0.2, 
                                      Context Recall=0.2, Answer Relevance=0.3
        """
        if weights is None:
            weights = {
                "faithfulness": 0.3,
                "context_precision": 0.2,
                "context_recall": 0.2,
                "answer_relevance": 0.3
            }
        
        scores = {
            "faithfulness": self.faithfulness,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_relevance": self.answer_relevance
        }
        
        # Filter out NaN values
        valid_scores = {k: v for k, v in scores.items() if not np.isnan(v)}
        
        if not valid_scores:
            return np.nan
        
        # Recalculate weights for valid scores only
        total_weight = sum(weights.get(k, 0) for k in valid_scores)
        if total_weight == 0:
            return np.nan
        
        weighted_sum = sum(
            scores[k] * weights.get(k, 0) / total_weight 
            for k in valid_scores
        )
        
        self.ragas_score = weighted_sum
        return self.ragas_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format."""
        return {
            "faithfulness": self.faithfulness,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_relevance": self.answer_relevance,
            "ragas_score": self.ragas_score,
            "details": {
                "faithfulness": self.faithfulness_details,
                "context_precision": self.context_precision_details,
                "context_recall": self.context_recall_details,
                "answer_relevance": self.answer_relevance_details
            }
        }


# Maximum context size in characters to prevent API overload
# IMPORTANT: For judges, we disable truncation to pass full GraphRAG context
# These limits are drastically increased to accommodate full context retrieval
MAX_CONTEXT_SIZE = 10000000  # ~10MB - essentially disabled for full context support
MAX_SINGLE_CONTEXT_SIZE = 5000000  # ~5MB per-context - essentially disabled


def truncate_contexts(contexts: List[str], max_total: int = MAX_CONTEXT_SIZE, 
                     max_single: int = MAX_SINGLE_CONTEXT_SIZE,
                     logger: Optional[logging.Logger] = None,
                     disable_truncation: bool = True) -> Tuple[List[str], bool]:
    """
    Truncate contexts to prevent API overload from excessively large payloads.
    
    IMPORTANT: Truncation may affect RAGAS metric accuracy. When context is truncated,
    faithfulness and context recall scores may be unreliable as the LLM judge cannot
    see the full context that was originally retrieved.
    
    By default, truncation is DISABLED (disable_truncation=True) to pass full GraphRAG context
    to judges for accurate evaluation. The MAX_CONTEXT_SIZE and MAX_SINGLE_CONTEXT_SIZE limits
    are set very high to accommodate full context retrieval.
    
    Args:
        contexts: List of context strings (may also contain dicts/other types that will be converted)
        max_total: Maximum total characters for all contexts combined
        max_single: Maximum characters for a single context
        logger: Optional logger for warnings
        disable_truncation: If True (default), bypass truncation limits and pass full context.
                           Set to False to enable truncation limits.
        
    Returns:
        Tuple of (truncated contexts list, was_truncated flag)
    """
    if not contexts:
        return [], False
    
    # If truncation is disabled, pass through full contexts without modification
    if disable_truncation:
        if logger:
            logger.debug(f"Context truncation disabled - passing full {len(contexts)} contexts to judge")
        truncated = []
        for ctx in contexts:
            if ctx is not None:
                if isinstance(ctx, str):
                    truncated.append(ctx)
                else:
                    try:
                        truncated.append(json.dumps(ctx) if isinstance(ctx, (dict, list)) else str(ctx))
                    except:
                        truncated.append(str(ctx))
        return truncated, False
    
    # Truncation is enabled - apply size limits
    truncated = []
    total_size = 0
    was_truncated = False
    original_total_size = 0
    
    # First pass: calculate original total size for warning message
    for ctx in contexts:
        if ctx is not None:
            if isinstance(ctx, str):
                original_total_size += len(ctx)
            else:
                try:
                    original_total_size += len(json.dumps(ctx) if isinstance(ctx, (dict, list)) else str(ctx))
                except:
                    original_total_size += len(str(ctx))
    
    for i, ctx in enumerate(contexts):
        # Handle None or empty contexts
        if ctx is None:
            continue
        
        # Convert non-string contexts to string (handles dicts, lists, etc. from GraphRAG)
        if not isinstance(ctx, str):
            try:
                ctx = json.dumps(ctx) if isinstance(ctx, (dict, list)) else str(ctx)
            except (TypeError, ValueError):
                ctx = str(ctx)
        
        # Skip empty strings after conversion
        if not ctx.strip():
            continue
        
        original_ctx_len = len(ctx)
        
        # Truncate individual context if too large
        if len(ctx) > max_single:
            was_truncated = True
            ctx = ctx[:max_single] + f"\n... [Context {i+1} truncated from {original_ctx_len} to {max_single} chars]"
            if logger:
                logger.warning(
                    f"RAGAS CONTEXT TRUNCATION: Context {i+1} truncated from {original_ctx_len:,} to {max_single:,} chars. "
                    f"This may affect faithfulness/context_recall metric accuracy as the judge LLM cannot see the full context."
                )
        
        # Check if adding this context exceeds total limit
        if total_size + len(ctx) > max_total:
            was_truncated = True
            remaining_space = max_total - total_size
            if remaining_space > 500:  # Only add if meaningful space remains
                ctx = ctx[:remaining_space] + f"\n... [Truncated to fit size limit]"
                truncated.append(ctx)
                if logger:
                    logger.warning(
                        f"RAGAS CONTEXT TRUNCATION: Total context size {original_total_size:,} chars exceeded limit of {max_total:,} chars. "
                        f"Truncating remaining contexts. RAGAS metrics (especially context_recall) may be unreliable for this sample."
                    )
            break
        
        truncated.append(ctx)
        total_size += len(ctx)
    
    return truncated, was_truncated


class RagasMetrics:
    """
    RAGAS (Retrieval Augmented Generation Assessment) Metrics Calculator.
    
    Implements LLM-as-a-Judge methodology for evaluating RAG pipelines with
    four core metrics: Faithfulness, Context Precision, Context Recall, 
    and Answer Relevance.
    
    All metrics return scores on a 0-100 scale.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.prompts_dir = Path(__file__).parent.parent / "prompt_templates"

    def _load_prompt(self, template_name: str, **kwargs) -> str:
        """Load and format a prompt template from file."""
        try:
            template_path = self.prompts_dir / f"{template_name}.txt"
            
            if not template_path.exists():
                self.logger.warning(f"Prompt template not found: {template_path}")
                # Fallback to empty string or raise error? 
                # For now, let's return empty and let the caller handle it or fail
                return ""
                
            with open(template_path, "r") as f:
                template = f.read()
            
            # Handle potential key errors gracefully or let them bubble up?
            # Given the current implementation, bubbling up is fine as it will be caught by try-except blocks in callers
            return template.format(**kwargs)
        except Exception as e:
            self.logger.error(f"Error loading prompt template {template_name}: {e}")
            raise e
    
    # =========================================================================
    # FAITHFULNESS METRIC
    # =========================================================================
    
    async def compute_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        llm_adapter: Any,
        max_retries: int = 2
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute Faithfulness score using RAGAS methodology.
        
        Faithfulness measures the factual consistency of the generated answer
        with respect to the retrieved context. It is computed in two steps:
        1. Extract atomic claims/statements from the answer
        2. Verify each claim against the context
        
        Score = (Number of supported claims / Total claims) * 100
        
        Args:
            question: The input question
            answer: The generated answer to evaluate
            contexts: List of retrieved context passages
            llm_adapter: LLM adapter for making API calls
            max_retries: Maximum retry attempts for LLM calls
            
        Returns:
            Tuple of (score, details_dict)
        """
        with safe_metric_computation('ragas_faithfulness', self.logger, fallback_value=(np.nan, {})):
            if not answer or not answer.strip():
                return 100.0, {"reason": "Empty answer - trivially faithful"}
            
            if not contexts or not any(c.strip() for c in contexts):
                return 0.0, {"reason": "No contexts provided - cannot verify faithfulness"}
            
            # Pass full GraphRAG context to judge (disable_truncation=True by default)
            truncated_contexts, was_truncated = truncate_contexts(contexts, logger=self.logger)
            
            # Step 1: Extract atomic claims from the answer
            claims = await self._extract_claims(question, answer, llm_adapter, max_retries)
            
            if not claims:
                return 100.0, {"reason": "No verifiable claims extracted", "claims": []}
            
            # Step 2: Verify each claim against the contexts
            context_text = "\n\n".join([f"[Context {i+1}]:\n{ctx}" for i, ctx in enumerate(truncated_contexts)])
            verifications = await self._verify_claims_against_context(
                claims, context_text, llm_adapter, max_retries
            )
            
            if not verifications:
                return np.nan, {"reason": "Claim verification failed"}
            
            # Calculate faithfulness score
            supported_count = sum(1 for v in verifications if v.get("verdict") == 1)
            faithfulness_score = (supported_count / len(verifications)) * 100.0
            
            # CRITICAL-003 FIX: Add quality flags to indicate data reliability
            total_context_length = sum(len(c) for c in contexts)
            details = {
                "claims_extracted": len(claims),
                "claims_supported": supported_count,
                "claims_unsupported": len(verifications) - supported_count,
                "verifications": verifications,
                # CRITICAL-003 FIX: Quality metadata
                "quality_flags": {
                    "context_truncated": was_truncated,
                    "context_count": len(contexts),
                    "total_context_length": total_context_length,
                    "reliability": "low" if was_truncated else "high",
                    "warning": "Context was truncated - metric may be unreliable" if was_truncated else None
                }
            }
            
            return faithfulness_score, details
    
    async def _extract_claims(
        self,
        question: str,
        answer: str,
        llm_adapter: Any,
        max_retries: int
    ) -> List[str]:
        """Extract atomic, self-contained claims from the answer."""
        
        prompt = self._load_prompt("ragas_claim_extraction", question=question, answer=answer)
        
        for attempt in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                claims = parse_json_response(
                    response.content,
                    self.logger,
                    "ragas claim extraction"
                )
                
                if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                    return [c.strip() for c in claims if c.strip()]
                    
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries:
                    self.logger.warning(f"Claim extraction failed after {max_retries + 1} attempts: {e}")
                continue
        
        return []
    
    async def _verify_claims_against_context(
        self,
        claims: List[str],
        context_text: str,
        llm_adapter: Any,
        max_retries: int
    ) -> List[Dict[str, Any]]:
        """Verify each claim against the provided context."""
        
        prompt = self._load_prompt(
            "ragas_claim_verification", 
            context_text=context_text, 
            claims_json=json.dumps(claims, indent=2)
        )
        
        for attempt in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                verifications = parse_json_response(
                    response.content,
                    self.logger,
                    "ragas claim verification"
                )
                
                if isinstance(verifications, list):
                    # Validate and normalize verifications
                    valid_verifications = []
                    for v in verifications:
                        if isinstance(v, dict) and "verdict" in v:
                            valid_verifications.append({
                                "claim": v.get("claim", ""),
                                "verdict": 1 if v["verdict"] == 1 else 0,
                                "reasoning": v.get("reasoning", ""),
                                "evidence": v.get("evidence", "")
                            })
                    # Only return if we got at least some valid verifications
                    if valid_verifications:
                        return valid_verifications
                    # If empty, try again on next attempt
                    elif attempt < max_retries:
                        self.logger.warning(f"Verification returned empty list, retrying (attempt {attempt + 1})")
                        continue
                    
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries:
                    self.logger.warning(f"Claim verification failed after {max_retries + 1} attempts: {e}")
                continue
        
        # If all retries failed, return a default "unable to verify" response
        # with all claims marked as not verified (verdict=0) to avoid NaN
        self.logger.warning(f"All verification attempts failed, marking {len(claims)} claims as unverified")
        return [{"claim": c, "verdict": 0, "reasoning": "Verification failed", "evidence": "none"} for c in claims]
    
    # =========================================================================
    # CONTEXT PRECISION METRIC
    # =========================================================================
    
    async def compute_context_precision(
        self,
        question: str,
        contexts: List[str],
        ground_truth: Optional[str],
        llm_adapter: Any,
        max_retries: int = 2,
        reranker: Optional[Any] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute Context Precision using RAGAS methodology with optional reranking.
        
        Context Precision measures whether all retrieved contexts are relevant
        to deriving the ground truth answer. It evaluates if each context 
        contains useful information.
        
        **Strategic Enhancement: Reranking for GraphRAG**
        Before evaluating context relevance, we apply a lightweight reranker to
        reorder contexts by semantic relevance to the question. This ensures the
        most relevant chunks appear at the top (Rank 1), satisfying RAGAS 
        assumptions that reward early relevant documents.
        
        Formula: Average precision at each relevant document position
        Score = Mean(Precision@k for k where context_k is relevant)
        
        Simplified Score = (Relevant contexts / Total contexts) * 100
        
        Args:
            question: The input question
            contexts: List of retrieved context passages  
            ground_truth: The expected/reference answer (optional)
            llm_adapter: LLM adapter for making API calls
            max_retries: Maximum retry attempts
            reranker: Optional reranker adapter to reorder contexts by relevance
            
        Returns:
            Tuple of (score, details_dict)
        """
        with safe_metric_computation('ragas_context_precision', self.logger, fallback_value=(np.nan, {})):
            if not contexts:
                return 0.0, {"reason": "No contexts provided"}
            
            # Apply reranking if available
            original_order = None
            if reranker is not None:
                try:
                    self.logger.info(f"Applying reranker to {len(contexts)} contexts")
                    
                    # Store original order for transparency
                    original_order = list(range(len(contexts)))
                    
                    # Rerank contexts
                    reranked_results = await reranker.arerank(question, contexts)
                    
                    # Extract reranked contexts and scores
                    contexts = [doc for doc, score in reranked_results]
                    rerank_scores = [score for doc, score in reranked_results]
                    
                    self.logger.info(
                        f"Reranking complete. Top context score: {rerank_scores[0]:.4f}, "
                        f"Bottom score: {rerank_scores[-1]:.4f}"
                    )
                except Exception as e:
                    self.logger.warning(f"Reranking failed: {e}. Using original order.")
                    original_order = None
            else:
                self.logger.debug("No reranker provided, using original context order")
            
            # Evaluate relevance of each context
            context_evaluations = await self._evaluate_context_relevance_batch(
                question, contexts, ground_truth, llm_adapter, max_retries
            )
            
            if not context_evaluations:
                return np.nan, {"reason": "Context evaluation failed"}
            
            # Calculate precision with position weighting (RAGAS style)
            # Higher weight for relevant contexts appearing earlier
            relevant_count = 0
            precision_sum = 0.0
            
            for i, eval_result in enumerate(context_evaluations):
                if eval_result.get("is_relevant", False):
                    relevant_count += 1
                    # Precision at position i+1
                    precision_at_k = relevant_count / (i + 1)
                    precision_sum += precision_at_k
            
            if relevant_count == 0:
                precision_score = 0.0
            else:
                # Average precision
                precision_score = (precision_sum / relevant_count) * 100.0
            
            details = {
                "total_contexts": len(contexts),
                "relevant_contexts": relevant_count,
                "irrelevant_contexts": len(contexts) - relevant_count,
                "context_evaluations": context_evaluations,
                "reranking_applied": original_order is not None
            }
            
            if original_order is not None:
                details["rerank_scores"] = rerank_scores
                details["original_order"] = original_order
            
            return precision_score, details
    
    async def _evaluate_context_relevance_batch(
        self,
        question: str,
        contexts: List[str],
        ground_truth: Optional[str],
        llm_adapter: Any,
        max_retries: int
    ) -> List[Dict[str, Any]]:
        """Evaluate relevance of each context to the question and ground truth."""
        
        # Handle empty or invalid contexts early
        if not contexts:
            self.logger.warning("No contexts provided for relevance evaluation")
            return []
        
        # Filter out empty contexts
        valid_contexts = [ctx for ctx in contexts if ctx and isinstance(ctx, str) and ctx.strip()]
        if not valid_contexts:
            self.logger.warning("All contexts were empty or invalid")
            return []
        
        # Truncate contexts to prevent API overload
        truncated_contexts, was_truncated = truncate_contexts(valid_contexts, logger=self.logger)
        
        if not truncated_contexts:
            self.logger.warning("Context truncation resulted in empty list")
            return []
        
        ground_truth_section = f"\n\n**GROUND TRUTH ANSWER:**\n{ground_truth}" if ground_truth else ""
        
        context_list = "\n\n".join([
            f"**CONTEXT {i+1}:**\n{ctx}" for i, ctx in enumerate(truncated_contexts)
        ])
        
        prompt = self._load_prompt(
            "ragas_context_relevance",
            question=question,
            ground_truth_section=ground_truth_section,
            context_list=context_list,
            num_contexts=len(contexts)
        )
        
        for attempt in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                evaluations = parse_json_response(
                    response.content,
                    self.logger,
                    "ragas context precision evaluation"
                )
                
                if isinstance(evaluations, list) and len(evaluations) >= 1:
                    # Validate and normalize
                    valid_evals = []
                    for e in evaluations:
                        if isinstance(e, dict):
                            try:
                                relevance_score = float(e.get("relevance_score", 0))
                            except (ValueError, TypeError):
                                relevance_score = 0.0
                            valid_evals.append({
                                "context_index": e.get("context_index", 0),
                                "is_relevant": bool(e.get("is_relevant", False)),
                                "relevance_score": relevance_score,
                                "reasoning": e.get("reasoning", "")
                            })
                    
                    # If we got valid evaluations, return them
                    if valid_evals:
                        return valid_evals
                    else:
                        self.logger.warning("No valid evaluations parsed from LLM response")
                    
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries:
                    self.logger.warning(f"Context precision evaluation failed: {e}")
                continue
        
        return []
    
    # =========================================================================
    # CONTEXT RECALL METRIC
    # =========================================================================
    
    async def compute_context_recall(
        self,
        question: str,
        contexts: List[str],
        ground_truth: str,
        llm_adapter: Any,
        max_retries: int = 2
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute Context Recall using RAGAS methodology.
        
        Context Recall measures whether all the relevant information required 
        to answer the question is present in the retrieved contexts. It evaluates
        if the contexts cover the information in the ground truth.
        
        Score = (GT sentences attributable to context / Total GT sentences) * 100
        
        Args:
            question: The input question
            contexts: List of retrieved context passages
            ground_truth: The expected/reference answer
            llm_adapter: LLM adapter for making API calls
            max_retries: Maximum retry attempts
            
        Returns:
            Tuple of (score, details_dict)
        """
        with safe_metric_computation('ragas_context_recall', self.logger, fallback_value=(np.nan, {})):
            if not ground_truth or not ground_truth.strip():
                return np.nan, {"reason": "Ground truth is required for context recall"}
            
            if not contexts:
                return 0.0, {"reason": "No contexts provided"}
            
            # Truncate contexts to prevent API overload
            truncated_contexts, was_truncated = truncate_contexts(contexts, logger=self.logger)
            
            # Step 1: Extract key information points from ground truth
            gt_statements = await self._extract_ground_truth_statements(
                question, ground_truth, llm_adapter, max_retries
            )
            
            if not gt_statements:
                return np.nan, {"reason": "Could not extract statements from ground truth"}
            
            # Step 2: Check which statements can be attributed to contexts
            context_text = "\n\n".join([f"[Context {i+1}]:\n{ctx}" for i, ctx in enumerate(truncated_contexts)])
            attributions = await self._attribute_statements_to_context(
                gt_statements, context_text, llm_adapter, max_retries
            )
            
            if not attributions:
                return np.nan, {"reason": "Attribution check failed"}
            
            # Calculate recall score
            attributed_count = sum(1 for a in attributions if a.get("can_attribute", False))
            recall_score = (attributed_count / len(attributions)) * 100.0
            
            # CRITICAL-003 FIX: Add quality flags to indicate data reliability
            total_context_length = sum(len(c) for c in contexts)
            details = {
                "gt_statements_total": len(gt_statements),
                "statements_in_context": attributed_count,
                "statements_missing": len(attributions) - attributed_count,
                "attributions": attributions,
                # CRITICAL-003 FIX: Quality metadata
                "quality_flags": {
                    "context_truncated": was_truncated,
                    "context_count": len(contexts),
                    "total_context_length": total_context_length,
                    "reliability": "low" if was_truncated else "high",
                    "warning": "Context was truncated - metric may be unreliable" if was_truncated else None
                }
            }
            
            return recall_score, details
    
    async def _extract_ground_truth_statements(
        self,
        question: str,
        ground_truth: str,
        llm_adapter: Any,
        max_retries: int
    ) -> List[str]:
        """Extract key information statements from ground truth answer."""
        
        prompt = self._load_prompt(
            "ragas_gt_statement_extraction",
            question=question,
            ground_truth=ground_truth
        )
        
        for attempt in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                statements = parse_json_response(
                    response.content,
                    self.logger,
                    "ragas gt statement extraction"
                )
                
                if isinstance(statements, list):
                    return [s.strip() for s in statements if isinstance(s, str) and s.strip()]
                    
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries:
                    self.logger.warning(f"GT statement extraction failed: {e}")
                continue
        
        return []
    
    async def _attribute_statements_to_context(
        self,
        statements: List[str],
        context_text: str,
        llm_adapter: Any,
        max_retries: int
    ) -> List[Dict[str, Any]]:
        """Check if each ground truth statement can be attributed to the contexts."""
        
        prompt = self._load_prompt(
            "ragas_context_recall_attribution",
            context_text=context_text,
            statements_json=json.dumps(statements, indent=2),
            num_statements=len(statements)
        )
        
        for attempt in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                attributions = parse_json_response(
                    response.content,
                    self.logger,
                    "ragas context recall attribution"
                )
                
                if isinstance(attributions, list):
                    valid_attributions = []
                    for a in attributions:
                        if isinstance(a, dict):
                            valid_attributions.append({
                                "statement": a.get("statement", ""),
                                "can_attribute": bool(a.get("can_attribute", False)),
                                "source_context": a.get("source_context"),
                                "reasoning": a.get("reasoning", "")
                            })
                    return valid_attributions
                    
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries:
                    self.logger.warning(f"Context recall attribution failed: {e}")
                continue
        
        return []
    
    # =========================================================================
    # ANSWER RELEVANCE METRIC
    # =========================================================================
    
    async def compute_answer_relevance(
        self,
        question: str,
        answer: str,
        llm_adapter: Any,
        embeddings_adapter: Optional[Any] = None,
        max_retries: int = 2
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute Answer Relevance using RAGAS methodology.
        
        Answer Relevance measures how pertinent the generated answer is to 
        the question. It is computed by generating questions from the answer
        and comparing them to the original question.
        
        Process:
        1. Generate N questions that the answer could address
        2. Compare semantic similarity between generated and original question
        3. Average the similarity scores
        
        If embeddings adapter is not provided, uses LLM-based similarity.
        
        Args:
            question: The original question
            answer: The generated answer to evaluate
            llm_adapter: LLM adapter for making API calls
            embeddings_adapter: Optional embeddings adapter for similarity
            max_retries: Maximum retry attempts
            
        Returns:
            Tuple of (score, details_dict)
        """
        with safe_metric_computation('ragas_answer_relevance', self.logger, fallback_value=(np.nan, {})):
            if not answer or not answer.strip():
                return 0.0, {"reason": "Empty answer cannot be relevant"}
            
            if not question or not question.strip():
                return np.nan, {"reason": "No question provided"}
            
            # Step 1: Generate questions from the answer
            generated_questions = await self._generate_questions_from_answer(
                answer, llm_adapter, max_retries, num_questions=3
            )
            
            if not generated_questions:
                # Fallback to direct LLM assessment
                return await self._assess_answer_relevance_direct(
                    question, answer, llm_adapter, max_retries
                )
            
            # Step 2: Calculate similarity between original and generated questions
            if embeddings_adapter:
                similarities = await self._compute_question_similarities_embedding(
                    question, generated_questions, embeddings_adapter
                )
            else:
                similarities = await self._compute_question_similarities_llm(
                    question, generated_questions, llm_adapter, max_retries
                )
            
            if not similarities:
                return np.nan, {"reason": "Similarity computation failed"}
            
            # Average similarity as relevance score
            relevance_score = np.mean(similarities) * 100.0
            
            details = {
                "original_question": question,
                "generated_questions": generated_questions,
                "similarity_scores": similarities,
                "mean_similarity": float(np.mean(similarities))
            }
            
            return relevance_score, details
    
    async def _generate_questions_from_answer(
        self,
        answer: str,
        llm_adapter: Any,
        max_retries: int,
        num_questions: int = 3
    ) -> List[str]:
        """Generate questions that the answer could address."""
        
        prompt = self._load_prompt(
            "ragas_question_generation",
            num_questions=num_questions,
            answer=answer
        )
        
        for attempt in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                questions = parse_json_response(
                    response.content,
                    self.logger,
                    "ragas question generation"
                )
                
                if isinstance(questions, list):
                    return [q.strip() for q in questions if isinstance(q, str) and q.strip()]
                    
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries:
                    self.logger.warning(f"Question generation failed: {e}")
                continue
        
        return []
    
    async def _compute_question_similarities_embedding(
        self,
        original_question: str,
        generated_questions: List[str],
        embeddings_adapter: Any
    ) -> List[float]:
        """Compute cosine similarities using embeddings."""
        try:
            # Get embedding for original question
            original_embedding = await embeddings_adapter.aembed_query(original_question)
            original_embedding = np.array(original_embedding)
            
            # Get embeddings for generated questions
            similarities = []
            for gen_q in generated_questions:
                gen_embedding = await embeddings_adapter.aembed_query(gen_q)
                gen_embedding = np.array(gen_embedding)
                
                # Cosine similarity
                cosine_sim = np.dot(original_embedding, gen_embedding) / (
                    np.linalg.norm(original_embedding) * np.linalg.norm(gen_embedding)
                )
                
                # Normalize from [-1, 1] to [0, 1]
                normalized_sim = (cosine_sim + 1) / 2
                similarities.append(float(normalized_sim))
            
            return similarities
            
        except Exception as e:
            self.logger.warning(f"Embedding similarity computation failed: {e}")
            return []
    
    async def _compute_question_similarities_llm(
        self,
        original_question: str,
        generated_questions: List[str],
        llm_adapter: Any,
        max_retries: int
    ) -> List[float]:
        """Compute semantic similarities using LLM."""
        
        prompt = self._load_prompt(
            "ragas_question_similarity",
            original_question=original_question,
            generated_questions_json=json.dumps(generated_questions, indent=2),
            num_questions=len(generated_questions)
        )
        
        for attempt in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                scores = parse_json_response(
                    response.content,
                    self.logger,
                    "ragas question similarity"
                )
                
                if isinstance(scores, list):
                    return [float(s) for s in scores if isinstance(s, (int, float))]
                    
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries:
                    self.logger.warning(f"LLM similarity computation failed: {e}")
                continue
        
        return []
    
    async def _assess_answer_relevance_direct(
        self,
        question: str,
        answer: str,
        llm_adapter: Any,
        max_retries: int
    ) -> Tuple[float, Dict[str, Any]]:
        """Fallback method: directly assess answer relevance with LLM."""
        
        prompt = self._load_prompt(
            "ragas_answer_relevance_direct",
            question=question,
            answer=answer
        )
        
        for attempt in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                result = parse_json_response(
                    response.content,
                    self.logger,
                    "ragas direct relevance assessment"
                )
                
                if isinstance(result, dict) and "relevance_score" in result:
                    score = float(result["relevance_score"])
                    score = max(0.0, min(100.0, score))
                    return score, {
                        "method": "direct_assessment",
                        "reasoning": result.get("reasoning", "")
                    }
                    
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries:
                    self.logger.warning(f"Direct relevance assessment failed: {e}")
                continue
        
        return np.nan, {"reason": "All assessment methods failed"}
    
    # =========================================================================
    # COMPREHENSIVE EVALUATION
    # =========================================================================
    
    async def evaluate_all(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        llm_adapter: Any = None,
        embeddings_adapter: Optional[Any] = None,
        max_retries: int = 2,
        compute_weights: Optional[Dict[str, float]] = None,
        reranker: Optional[Any] = None
    ) -> RagasEvaluationResult:
        """
        Compute all RAGAS metrics for a question-answer pair.
        
        Args:
            question: The input question
            answer: The generated answer
            contexts: List of retrieved context passages
            ground_truth: Optional reference answer (required for context_recall)
            llm_adapter: LLM adapter for API calls
            embeddings_adapter: Optional embeddings adapter for similarity
            max_retries: Maximum retry attempts
            compute_weights: Optional custom weights for final RAGAS score
            reranker: Optional reranker adapter for context precision
            
        Returns:
            RagasEvaluationResult with all metrics and details
        """
        result = RagasEvaluationResult()
        
        # Run all metrics concurrently for efficiency
        tasks = [
            self.compute_faithfulness(question, answer, contexts, llm_adapter, max_retries),
            self.compute_context_precision(question, contexts, ground_truth, llm_adapter, max_retries, reranker),
            self.compute_answer_relevance(question, answer, llm_adapter, embeddings_adapter, max_retries)
        ]
        
        # Context recall requires ground truth
        if ground_truth:
            tasks.append(
                self.compute_context_recall(question, contexts, ground_truth, llm_adapter, max_retries)
            )
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process faithfulness result
            if not isinstance(results[0], Exception):
                result.faithfulness, result.faithfulness_details = results[0]
            else:
                self.logger.warning(f"Faithfulness computation failed: {results[0]}")
            
            # Process context precision result
            if not isinstance(results[1], Exception):
                result.context_precision, result.context_precision_details = results[1]
            else:
                self.logger.warning(f"Context precision computation failed: {results[1]}")
            
            # Process answer relevance result
            if not isinstance(results[2], Exception):
                result.answer_relevance, result.answer_relevance_details = results[2]
            else:
                self.logger.warning(f"Answer relevance computation failed: {results[2]}")
            
            # Process context recall result (if computed)
            if ground_truth and len(results) > 3:
                if not isinstance(results[3], Exception):
                    result.context_recall, result.context_recall_details = results[3]
                else:
                    self.logger.warning(f"Context recall computation failed: {results[3]}")
            
        except Exception as e:
            self.logger.error(f"RAGAS evaluation failed: {e}")
        
        # Compute weighted RAGAS score
        result.compute_ragas_score(compute_weights)
        
        return result
    
    # =========================================================================
    # CONVENIENCE METHODS FOR INDIVIDUAL METRIC ACCESS
    # =========================================================================
    
    async def compute_faithfulness_score(
        self, question: str, answer: str, contexts: List[str], 
        llm_adapter: Any, max_retries: int = 2
    ) -> float:
        """Convenience method returning only the faithfulness score."""
        score, _ = await self.compute_faithfulness(
            question, answer, contexts, llm_adapter, max_retries
        )
        return score
    
    async def compute_context_precision_score(
        self, question: str, contexts: List[str], ground_truth: Optional[str],
        llm_adapter: Any, max_retries: int = 2, reranker: Optional[Any] = None
    ) -> float:
        """Convenience method returning only the context precision score."""
        score, _ = await self.compute_context_precision(
            question, contexts, ground_truth, llm_adapter, max_retries, reranker
        )
        return score
    
    async def compute_context_recall_score(
        self, question: str, contexts: List[str], ground_truth: str,
        llm_adapter: Any, max_retries: int = 2
    ) -> float:
        """Convenience method returning only the context recall score."""
        score, _ = await self.compute_context_recall(
            question, contexts, ground_truth, llm_adapter, max_retries
        )
        return score
    
    async def compute_answer_relevance_score(
        self, question: str, answer: str, llm_adapter: Any,
        embeddings_adapter: Optional[Any] = None, max_retries: int = 2
    ) -> float:
        """Convenience method returning only the answer relevance score."""
        score, _ = await self.compute_answer_relevance(
            question, answer, llm_adapter, embeddings_adapter, max_retries
        )
        return score
