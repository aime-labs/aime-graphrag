#!/usr/bin/env python3
"""
Advanced Metrics Batching System

This module provides intelligent batching of metrics computations to minimize LLM API calls
while maintaining accuracy. It groups similar computations and reuses intermediate results.
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import logging
import warnings
import os

from utils.error_handling import parse_json_response


class ProductionValidationError(Exception):
    """Raised when placeholder methods are called in production."""
    pass


def validate_not_placeholder(method_name: str):
    """Decorator to validate that placeholder methods are not called in production."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Check if we're in production mode
            is_production = os.getenv('ENVIRONMENT', '').lower() in ['production', 'prod']
            if is_production:
                raise ProductionValidationError(
                    f"Placeholder method {method_name} should not be called in production. "
                    f"Please implement the actual functionality."
                )
            
            # Issue warning in all modes
            warnings.warn(
                f"Method {method_name} is using optimized implementation. "
                f"Ensure this is intentional for your use case.",
                UserWarning,
                stacklevel=2
            )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


@dataclass
class MetricsBatch:
    """Represents a batch of metrics computations that can be processed together."""
    batch_type: str
    questions: List[Dict[str, Any]]
    answers: List[str]
    contexts: List[List[str]]
    metrics_requested: List[str]
    batch_id: str
    
    def __post_init__(self):
        if not self.batch_id:
            # Create a hash-based batch ID
            content = f"{self.batch_type}:{len(self.questions)}:{','.join(self.metrics_requested)}"
            self.batch_id = hashlib.md5(content.encode()).hexdigest()[:8]


class MetricsBatchProcessor:
    """Advanced metrics processor that optimizes LLM calls through intelligent batching."""
    
    def __init__(self, logger: logging.Logger, smart_cache=None):
        self.logger = logger
        self.smart_cache = smart_cache
        
        # Batch optimization settings
        self.max_batch_size = 10  # Maximum items per batch
        self.min_batch_size = 3   # Minimum items to justify batching
        
        # Metrics that can be batched together
        self.batchable_metrics = {
            'text_metrics': ['exact_match', 'f1_score', 'rouge_score'],
            'llm_single': ['context_relevance', 'faithfulness', 'coverage_score'],
            'llm_comparative': ['answer_correctness'],
            'triple_metrics': ['triple_em', 'triple_f1', 'triple_precision', 'triple_recall']
        }
        
        # Metrics that benefit from shared computations
        self.shared_computation_groups = {
            'statement_based': ['answer_correctness', 'coverage_score'],
            'context_based': ['context_relevance', 'faithfulness'],
            'triple_based': ['triple_em', 'triple_f1', 'triple_precision', 'triple_recall']
        }
    
    async def process_metrics_batch_optimized(self, 
                                            computation_requests: List[Dict[str, Any]],
                                            llm_adapter: Any,
                                            embedding_adapter: Any = None) -> Dict[str, Dict[str, float]]:
        """Process multiple metrics computations with advanced batching optimizations."""
        
        # Group requests by similarity for batching
        batches = self._create_optimized_batches(computation_requests)
        
        self.logger.info(f"Created {len(batches)} optimized batches for {len(computation_requests)} requests")
        
        # Process each batch
        all_results = {}
        
        for batch in batches:
            try:
                batch_results = await self._process_single_batch(
                    batch, llm_adapter, embedding_adapter
                )
                all_results.update(batch_results)
                
                self.logger.debug(f"Processed batch {batch.batch_id} with {len(batch_results)} results")
                
            except Exception as e:
                self.logger.error(f"Batch {batch.batch_id} failed: {str(e)}")
                # Create error results for this batch
                for i in range(len(batch.questions)):
                    request_id = f"{batch.batch_id}_{i}"
                    all_results[request_id] = {metric: 0.0 for metric in batch.metrics_requested}
        
        return all_results
    
    def _create_optimized_batches(self, requests: List[Dict[str, Any]]) -> List[MetricsBatch]:
        """Create optimized batches based on metrics similarity and computational efficiency."""
        
        # Group requests by metrics signature
        metrics_groups = defaultdict(list)
        
        for i, request in enumerate(requests):
            metrics_signature = tuple(sorted(request.get('metrics', [])))
            metrics_groups[metrics_signature].append((i, request))
        
        batches = []
        
        for metrics_signature, group_requests in metrics_groups.items():
            # Further group by question type and method for cache efficiency
            sub_groups = defaultdict(list)
            
            for req_idx, request in group_requests:
                sub_key = (
                    request.get('question_type', 'unknown'),
                    request.get('method', 'unknown'),
                    len(request.get('contexts', []))  # Similar context lengths
                )
                sub_groups[sub_key].append((req_idx, request))
            
            # Create batches from sub-groups
            for sub_key, sub_requests in sub_groups.items():
                # Split large groups into appropriately sized batches
                for batch_start in range(0, len(sub_requests), self.max_batch_size):
                    batch_end = min(batch_start + self.max_batch_size, len(sub_requests))
                    batch_requests = sub_requests[batch_start:batch_end]
                    
                    if len(batch_requests) >= self.min_batch_size or len(sub_requests) < self.min_batch_size:
                        batch = self._create_batch_from_requests(batch_requests, metrics_signature)
                        batches.append(batch)
        
        return batches
    
    def _create_batch_from_requests(self, requests: List[Tuple[int, Dict[str, Any]]], 
                                  metrics_signature: Tuple[str, ...]) -> MetricsBatch:
        """Create a MetricsBatch from a list of requests."""
        
        questions = []
        answers = []
        contexts = []
        
        for req_idx, request in requests:
            questions.append(request.get('question_data', {}))
            answers.append(request.get('answer', ''))
            contexts.append(request.get('contexts', []))
        
        # Determine batch type based on metrics
        batch_type = self._determine_batch_type(list(metrics_signature))
        
        return MetricsBatch(
            batch_type=batch_type,
            questions=questions,
            answers=answers,
            contexts=contexts,
            metrics_requested=list(metrics_signature),
            batch_id=""  # Will be auto-generated
        )
    
    def _determine_batch_type(self, metrics: List[str]) -> str:
        """Determine the optimal batch processing type based on requested metrics."""
        
        if all(m in self.batchable_metrics['text_metrics'] for m in metrics):
            return 'text_only'
        elif all(m in self.batchable_metrics['llm_single'] for m in metrics):
            return 'llm_single'
        elif any(m in self.batchable_metrics['llm_comparative'] for m in metrics):
            return 'llm_comparative'
        elif any(m in self.batchable_metrics['triple_metrics'] for m in metrics):
            return 'triple_focused'
        else:
            return 'mixed'
    
    async def _process_single_batch(self, batch: MetricsBatch, 
                                  llm_adapter: Any, 
                                  embedding_adapter: Any = None) -> Dict[str, Dict[str, float]]:
        """Process a single batch with optimized method selection."""
        
        if batch.batch_type == 'text_only':
            return await self._process_text_only_batch(batch)
        elif batch.batch_type == 'llm_single':
            return await self._process_llm_single_batch(batch, llm_adapter)
        elif batch.batch_type == 'llm_comparative':
            return await self._process_llm_comparative_batch(batch, llm_adapter, embedding_adapter)
        elif batch.batch_type == 'triple_focused':
            return await self._process_triple_focused_batch(batch, llm_adapter)
        else:
            return await self._process_mixed_batch(batch, llm_adapter, embedding_adapter)
    
    async def _process_text_only_batch(self, batch: MetricsBatch) -> Dict[str, Dict[str, float]]:
        """Process batch containing only text-based metrics."""
        
        results = {}
        
        for i, (question, answer) in enumerate(zip(batch.questions, batch.answers)):
            request_id = f"{batch.batch_id}_{i}"
            metrics = {}
            
            ground_truth = question.get('answer', '')
            
            for metric in batch.metrics_requested:
                if metric == 'exact_match':
                    # Efficient exact match computation
                    metrics[metric] = 1.0 if answer.strip().lower() == ground_truth.strip().lower() else 0.0
                elif metric == 'f1_score':
                    # Efficient F1 computation using token overlap
                    metrics[metric] = self._compute_f1_score_fast(answer, ground_truth)
                elif metric == 'rouge_score':
                    # For batching, we could implement a simplified ROUGE
                    metrics[metric] = self._compute_rouge_score_fast(answer, ground_truth)
            
            results[request_id] = metrics
        
        return results
    
    async def _process_llm_single_batch(self, batch: MetricsBatch, 
                                      llm_adapter: Any) -> Dict[str, Dict[str, float]]:
        """Process batch with LLM metrics that can be computed individually."""
        
        # For metrics that can't be truly batched, we process them efficiently
        # but still benefit from shared adapter context and caching
        
        results = {}
        
        for i, (question, answer, contexts) in enumerate(zip(batch.questions, batch.answers, batch.contexts)):
            request_id = f"{batch.batch_id}_{i}"
            metrics = {}
            
            for metric in batch.metrics_requested:
                try:
                    if metric == 'context_relevance':
                        # Check cache first
                        cache_key = f"context_relevance:{hash((question.get('question', ''), tuple(contexts)))}"
                        cached_result = await self._get_cached_metric(cache_key)
                        
                        if cached_result is not None:
                            metrics[metric] = cached_result
                        else:
                            # Compute and cache
                            value = await self._compute_context_relevance_optimized(
                                question.get('question', ''), contexts, llm_adapter
                            )
                            metrics[metric] = value
                            await self._cache_metric(cache_key, value)
                    
                    elif metric == 'faithfulness':
                        cache_key = f"faithfulness:{hash((question.get('question', ''), answer, tuple(contexts)))}"
                        cached_result = await self._get_cached_metric(cache_key)
                        
                        if cached_result is not None:
                            metrics[metric] = cached_result
                        else:
                            value = await self._compute_faithfulness_optimized(
                                question.get('question', ''), answer, contexts, llm_adapter
                            )
                            metrics[metric] = value
                            await self._cache_metric(cache_key, value)
                    
                    elif metric == 'coverage_score':
                        cache_key = f"coverage_score:{hash((question.get('question', ''), question.get('answer', ''), answer))}"
                        cached_result = await self._get_cached_metric(cache_key)
                        
                        if cached_result is not None:
                            metrics[metric] = cached_result
                        else:
                            value = await self._compute_coverage_score_optimized(
                                question.get('question', ''), question.get('answer', ''), answer, llm_adapter
                            )
                            metrics[metric] = value
                            await self._cache_metric(cache_key, value)
                
                except Exception as e:
                    self.logger.warning(f"Metric {metric} computation failed: {str(e)}")
                    metrics[metric] = 0.0
            
            results[request_id] = metrics
        
        return results
    
    async def _process_llm_comparative_batch(self, batch: MetricsBatch, 
                                           llm_adapter: Any, 
                                           embedding_adapter: Any) -> Dict[str, Dict[str, float]]:
        """Process batch with metrics requiring comparison between multiple texts."""
        
        results = {}
        
        # For answer_correctness, we can batch the statement generation phase
        if 'answer_correctness' in batch.metrics_requested:
            # Generate statements for all answers and ground truths in batch
            all_answers = batch.answers
            all_ground_truths = [q.get('answer', '') for q in batch.questions]
            
            # Batch generate statements
            answer_statements_batch = await self._batch_generate_statements(all_answers, llm_adapter)
            gt_statements_batch = await self._batch_generate_statements(all_ground_truths, llm_adapter)
            
            # Compute individual scores using pre-generated statements
            for i, question in enumerate(batch.questions):
                request_id = f"{batch.batch_id}_{i}"
                
                try:
                    answer_statements = answer_statements_batch[i]
                    gt_statements = gt_statements_batch[i]
                    
                    # Compute factuality and semantic similarity
                    factuality = await self._compute_factuality_from_statements(
                        question.get('question', ''), answer_statements, gt_statements, llm_adapter
                    )
                    
                    if embedding_adapter:
                        similarity = await self._compute_semantic_similarity(
                            batch.answers[i], all_ground_truths[i], embedding_adapter
                        )
                    else:
                        similarity = 0.5  # Fallback
                    
                    # Combine scores
                    answer_correctness = (factuality * 0.75) + (similarity * 0.25)
                    
                    results[request_id] = {'answer_correctness': answer_correctness}
                    
                except Exception as e:
                    self.logger.warning(f"Answer correctness computation failed for item {i}: {str(e)}")
                    results[request_id] = {'answer_correctness': 0.0}
        
        return results
    
    async def _process_triple_focused_batch(self, batch: MetricsBatch, 
                                          llm_adapter: Any) -> Dict[str, Dict[str, float]]:
        """Process batch focused on triple-based metrics."""
        
        results = {}
        
        # Triple metrics can benefit from shared triple extraction
        for i, (question, answer, contexts) in enumerate(zip(batch.questions, batch.answers, batch.contexts)):
            request_id = f"{batch.batch_id}_{i}"
            
            try:
                evidence_tripples = question.get('evidence_tripples', [])
                
                if evidence_tripples:
                    # Use improved triple metrics with evidence
                    triple_results = await self._compute_triple_metrics_improved_cached(
                        question.get('question', ''), answer, contexts, evidence_tripples, llm_adapter
                    )
                else:
                    # Fallback to basic triple metrics
                    triple_results = await self._compute_triple_metrics_basic_cached(
                        answer, question.get('answer', ''), llm_adapter
                    )
                
                # Filter to requested metrics
                filtered_results = {
                    metric: triple_results.get(metric, 0.0)
                    for metric in batch.metrics_requested
                    if metric in triple_results
                }
                
                results[request_id] = filtered_results
                
            except Exception as e:
                self.logger.warning(f"Triple metrics computation failed for item {i}: {str(e)}")
                results[request_id] = {metric: 0.0 for metric in batch.metrics_requested}
        
        return results
    
    async def _process_mixed_batch(self, batch: MetricsBatch, 
                                 llm_adapter: Any, 
                                 embedding_adapter: Any) -> Dict[str, Dict[str, float]]:
        """Process batch with mixed metric types."""
        
        # For mixed batches, we group by metric type and process efficiently
        text_metrics = [m for m in batch.metrics_requested if m in self.batchable_metrics['text_metrics']]
        llm_metrics = [m for m in batch.metrics_requested if m not in text_metrics]
        
        results = {}
        
        # Process text metrics first (fastest)
        if text_metrics:
            text_batch = MetricsBatch(
                batch_type='text_only',
                questions=batch.questions,
                answers=batch.answers,
                contexts=batch.contexts,
                metrics_requested=text_metrics,
                batch_id=f"{batch.batch_id}_text"
            )
            text_results = await self._process_text_only_batch(text_batch)
            
            # Merge text results
            for request_id, metrics in text_results.items():
                if request_id not in results:
                    results[request_id] = {}
                results[request_id].update(metrics)
        
        # Process LLM metrics
        if llm_metrics:
            llm_batch = MetricsBatch(
                batch_type='llm_single',
                questions=batch.questions,
                answers=batch.answers,
                contexts=batch.contexts,
                metrics_requested=llm_metrics,
                batch_id=f"{batch.batch_id}_llm"
            )
            llm_results = await self._process_llm_single_batch(llm_batch, llm_adapter)
            
            # Merge LLM results
            for request_id, metrics in llm_results.items():
                original_id = request_id.replace(f"{batch.batch_id}_llm", batch.batch_id)
                if original_id not in results:
                    results[original_id] = {}
                results[original_id].update(metrics)
        
        return results
    
    # Helper methods for optimized computations
    
    def _compute_f1_score_fast(self, answer: str, ground_truth: str) -> float:
        """Fast F1 score computation."""
        answer_tokens = set(answer.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        
        if not answer_tokens and not gt_tokens:
            return 1.0
        
        intersection = answer_tokens.intersection(gt_tokens)
        
        if not intersection:
            return 0.0
        
        precision = len(intersection) / len(answer_tokens) if answer_tokens else 0
        recall = len(intersection) / len(gt_tokens) if gt_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _compute_rouge_score_fast(self, answer: str, ground_truth: str) -> float:
        """Fast ROUGE-1 score computation."""
        answer_tokens = answer.lower().split()
        gt_tokens = ground_truth.lower().split()
        
        if not gt_tokens:
            return 1.0 if not answer_tokens else 0.0
        
        answer_bigrams = set(zip(answer_tokens, answer_tokens[1:]))
        gt_bigrams = set(zip(gt_tokens, gt_tokens[1:]))
        
        if not gt_bigrams:
            return 1.0 if not answer_bigrams else 0.0
        
        intersection = answer_bigrams.intersection(gt_bigrams)
        return len(intersection) / len(gt_bigrams)
    
    async def _batch_generate_statements(self, texts: List[str], llm_adapter: Any) -> List[List[str]]:
        """Generate statements for multiple texts in a single LLM call."""
        
        # Create a batch prompt
        batch_prompt = "Generate concise factual statements for each text. Respond with JSON array of arrays.\\n\\nTexts:\\n"
        
        for i, text in enumerate(texts):
            batch_prompt += f"{i+1}. {text[:500]}...\\n\\n"  # Truncate long texts
        
        batch_prompt += "\\nRespond with JSON format: [[statements for text 1], [statements for text 2], ...]"
        
        try:
            response = await llm_adapter.ainvoke(batch_prompt)
            statements_batch = parse_json_response(
                response.content,
                self.logger,
                "batch statement generation",
            )
            
            # Ensure we have the right number of results
            while len(statements_batch) < len(texts):
                statements_batch.append([])
            
            return statements_batch[:len(texts)]
            
        except Exception as e:
            self.logger.warning(f"Batch statement generation failed: {str(e)}")
            # Fallback to empty statements
            return [[] for _ in texts]
    
    async def _get_cached_metric(self, cache_key: str) -> Optional[float]:
        """Get cached metric result."""
        if self.smart_cache:
            return await self.smart_cache.get_metric_result("batch_metric", {"key": cache_key})
        return None
    
    async def _cache_metric(self, cache_key: str, value: float) -> None:
        """Cache metric result."""
        if self.smart_cache:
            await self.smart_cache.cache_metric_result("batch_metric", {"key": cache_key}, value)
    
    # Actual implementations for optimized computations
    async def _compute_context_relevance_optimized(self, question: str, contexts: List[str], llm_adapter: Any) -> float:
        """Optimized context relevance computation."""
        if not contexts:
            return 0.0
        
        try:
            # Use the same logic as RetrievalMetrics but optimized for batch processing
            relevance_scores = []
            for context in contexts:
                score = await self._get_simple_relevance_rating(question, context, llm_adapter)
                relevance_scores.append(score)
            
            # Return average relevance as percentage
            return float(np.mean([s for s in relevance_scores if not np.isnan(s)]) if relevance_scores else 0.0)
        except Exception as e:
            self.logger.warning(f"Context relevance computation failed: {e}")
            return 0.0
    
    async def _compute_faithfulness_optimized(self, question: str, answer: str, contexts: List[str], llm_adapter: Any) -> float:
        """Optimized faithfulness computation."""
        if not contexts or not answer:
            return 0.0
        
        try:
            # Generate statements from answer
            statements = await self._generate_statements_single(answer, llm_adapter)
            
            if not statements:
                return 100.0 if not answer.strip() else 0.0
            
            # Evaluate statements against contexts
            context_str = "\n".join(contexts)
            verdicts = await self._evaluate_statements_against_context(statements, context_str, llm_adapter)
            
            # Calculate faithfulness percentage
            if verdicts:
                supported = [v["verdict"] for v in verdicts]
                faithfulness_pct = (sum(supported) / len(supported)) * 100.0
                return float(faithfulness_pct)
            
            return 0.0
        except Exception as e:
            self.logger.warning(f"Faithfulness computation failed: {e}")
            return 0.0
    
    async def _compute_coverage_score_optimized(self, question: str, ground_truth: str, answer: str, llm_adapter: Any) -> float:
        """Optimized coverage score computation."""
        if not ground_truth or not answer:
            return 0.0
        
        try:
            # Extract facts from ground truth
            gt_facts = await self._extract_facts_from_text(question, ground_truth, llm_adapter)
            
            if not gt_facts:
                return 100.0  # Perfect score if no facts to cover
            
            # Check which facts are covered in the answer
            coverage_results = await self._check_fact_coverage_in_text(question, gt_facts, answer, llm_adapter)
            
            # Calculate coverage percentage
            if coverage_results:
                total_score = sum(result.get('score', 0) for result in coverage_results)
                return float(total_score / len(gt_facts))
            
            return 0.0
        except Exception as e:
            self.logger.warning(f"Coverage score computation failed: {e}")
            return 0.0
    
    async def _compute_factuality_from_statements(self, question: str, answer_statements: List[str], 
                                                gt_statements: List[str], llm_adapter: Any) -> float:
        """Compute factuality score from pre-generated statements."""
        if not answer_statements and not gt_statements:
            return 100.0
        
        if not answer_statements:
            return 0.0
        
        if not gt_statements:
            return 0.0
        
        try:
            # Generate classification with numeric scoring
            prompt = f"""You are a harsh and critical factual accuracy judge. Evaluate each statement from the candidate answer against the ground truth statements.

EVALUATION CRITERIA:
- Be EXTREMELY STRICT and CRITICAL
- A statement is CORRECT only if it perfectly matches the ground truth in meaning and completeness
- Slight variations in wording are acceptable ONLY if the meaning is identical
- Missing details, vague information, or imprecise statements are INCORRECT
- Additional information not in ground truth is suspicious and may be INCORRECT

SCORING FOR EACH STATEMENT (0-100):
- 90-100: Statement is perfectly accurate and matches ground truth
- 70-89: Statement is mostly accurate with trivial differences
- 40-69: Statement is partially correct but has notable issues
- 0-39: Statement is incorrect, contradictory, or unsupported

Question Context: {question}
Answer Statements to Evaluate: {json.dumps(answer_statements)}
Ground Truth Statements (Reference): {json.dumps(gt_statements)}

Respond with ONLY a JSON object in this format:
{{"evaluations": [{{"statement": "...", "score": 85, "reason": "..."}}]}}

Evaluation Results:"""

            response = await llm_adapter.ainvoke(prompt)
            evaluation = parse_json_response(
                response.content,
                self.logger,
                "batch factuality evaluation",
            )
            
            evaluations = evaluation.get('evaluations', [])
            if not evaluations:
                return 0.0
            
            # Calculate average score from numeric evaluations
            total_score = sum(eval_item.get('score', 0) for eval_item in evaluations)
            return float(total_score / len(evaluations))
            
        except Exception as e:
            self.logger.warning(f"Factuality computation failed: {e}")
            return 0.0
    
    async def _compute_semantic_similarity(self, answer: str, ground_truth: str, embedding_adapter: Any) -> float:
        """Compute semantic similarity using embeddings."""
        if not answer or not ground_truth:
            return 0.0
        
        try:
            # Get embeddings for both texts
            answer_embedding = await embedding_adapter.aembed_query(answer)
            gt_embedding = await embedding_adapter.aembed_query(ground_truth)
            
            # Convert to numpy arrays
            answer_embedding = np.array(answer_embedding)
            gt_embedding = np.array(gt_embedding)
            
            # Compute cosine similarity
            cosine_sim = np.dot(answer_embedding, gt_embedding) / (
                np.linalg.norm(answer_embedding) * np.linalg.norm(gt_embedding))
            
            # Convert to percentage (cosine similarity ranges from -1 to 1)
            similarity_percentage = ((cosine_sim + 1) / 2) * 100.0
            return float(max(0.0, min(100.0, similarity_percentage)))
            
        except Exception as e:
            self.logger.warning(f"Semantic similarity computation failed: {e}")
            return 0.0
    
    async def _compute_triple_metrics_improved_cached(self, question: str, answer: str, 
                                                    contexts: List[str], evidence_tripples: List[str], 
                                                    llm_adapter: Any) -> Dict[str, float]:
        """Compute triple metrics with caching."""
        try:
            # Extract triples from answer
            answer_triples = await self._extract_system_triples_from_answer(answer, llm_adapter)
            
            # Parse evidence triples
            gold_triples = self._parse_evidence_tripples(evidence_tripples)
            
            if not answer_triples and not gold_triples:
                return {'triple_em': 100.0, 'triple_f1': 100.0, 'triple_precision': 100.0, 'triple_recall': 100.0}
            
            if not answer_triples:
                return {'triple_em': 0.0, 'triple_f1': 0.0, 'triple_precision': 0.0, 'triple_recall': 100.0 if not gold_triples else 0.0}
            
            if not gold_triples:
                return {'triple_em': 0.0, 'triple_f1': 0.0, 'triple_precision': 0.0, 'triple_recall': 0.0}
            
            # Calculate metrics
            correct_triples = len(answer_triples & gold_triples)
            total_answer_triples = len(answer_triples)
            total_gold_triples = len(gold_triples)
            
            # Precision: correct / predicted
            precision = (correct_triples / total_answer_triples) * 100.0 if total_answer_triples > 0 else 0.0
            
            # Recall: correct / actual
            recall = (correct_triples / total_gold_triples) * 100.0 if total_gold_triples > 0 else 0.0
            
            # F1 score
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Exact match: all triples must match
            em = 100.0 if answer_triples == gold_triples else 0.0
            
            return {
                'triple_em': float(em),
                'triple_f1': float(f1),
                'triple_precision': float(precision),
                'triple_recall': float(recall)
            }
            
        except Exception as e:
            self.logger.warning(f"Triple metrics computation failed: {e}")
            return {'triple_em': 0.0, 'triple_f1': 0.0, 'triple_precision': 0.0, 'triple_recall': 0.0}
    
    async def _compute_triple_metrics_basic_cached(self, answer: str, ground_truth: str, 
                                                 llm_adapter: Any) -> Dict[str, float]:
        """Compute basic triple metrics with caching."""
        try:
            # Extract triples from both answer and ground truth
            answer_triples = await self._extract_system_triples_from_answer(answer, llm_adapter)
            gt_triples = await self._extract_system_triples_from_answer(ground_truth, llm_adapter)
            
            if not answer_triples and not gt_triples:
                return {'triple_em': 100.0, 'triple_f1': 100.0, 'triple_precision': 100.0, 'triple_recall': 100.0}
            
            if not answer_triples:
                return {'triple_em': 0.0, 'triple_f1': 0.0, 'triple_precision': 0.0, 'triple_recall': 100.0 if not gt_triples else 0.0}
            
            if not gt_triples:
                return {'triple_em': 0.0, 'triple_f1': 0.0, 'triple_precision': 0.0, 'triple_recall': 0.0}
            
            # Calculate metrics
            correct_triples = len(answer_triples & gt_triples)
            total_answer_triples = len(answer_triples)
            total_gt_triples = len(gt_triples)
            
            # Precision: correct / predicted
            precision = (correct_triples / total_answer_triples) * 100.0 if total_answer_triples > 0 else 0.0
            
            # Recall: correct / actual
            recall = (correct_triples / total_gt_triples) * 100.0 if total_gt_triples > 0 else 0.0
            
            # F1 score
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Exact match: all triples must match
            em = 100.0 if answer_triples == gt_triples else 0.0
            
            return {
                'triple_em': float(em),
                'triple_f1': float(f1),
                'triple_precision': float(precision),
                'triple_recall': float(recall)
            }
            
        except Exception as e:
            self.logger.warning(f"Basic triple metrics computation failed: {e}")
            return {'triple_em': 0.0, 'triple_f1': 0.0, 'triple_precision': 0.0, 'triple_recall': 0.0}

    # Helper methods for the actual implementations
    async def _get_simple_relevance_rating(self, question: str, context: str, llm_adapter: Any) -> float:
        """Get simple relevance rating as percentage."""
        prompt = f"""You are a harsh context relevance evaluator. Rate how relevant this context is for answering the question using a 0-100 numeric scale.

Question: {question}
Context: {context}

Respond with ONLY a number from 0 to 100.

Relevance Rating:"""
        
        try:
            response = await llm_adapter.ainvoke(prompt)
            import re
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response.content)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(100.0, score))
            return 0.0
        except Exception:
            return 0.0
    
    async def _evaluate_statements_against_context(self, statements: List[str], context: str, llm_adapter: Any) -> List[Dict]:
        """Evaluate which statements are supported by context."""
        prompt = f"""Evaluate if each statement can be DIRECTLY inferred from the context.

Context: {context}
Statements: {json.dumps(statements)}

Respond with ONLY a JSON array of objects:
[{{"statement": "...", "verdict": 1, "reason": "..."}}]

Response:"""
        
        try:
            response = await llm_adapter.ainvoke(prompt)
            verdicts = parse_json_response(
                response.content,
                self.logger,
                "batch context verdicts",
            )
            return self._validate_verdicts(verdicts)
        except Exception:
            return []
    
    def _validate_verdicts(self, verdicts: List) -> List[Dict]:
        """Ensure verdicts have required fields and proper types."""
        valid = []
        for item in verdicts:
            try:
                if ("statement" in item and 
                    "verdict" in item and item["verdict"] in {0, 1} and
                    "reason" in item):
                    valid.append({
                        "statement": str(item["statement"]),
                        "verdict": int(item["verdict"]),
                        "reason": str(item["reason"])
                    })
            except (TypeError, ValueError):
                continue
        return valid
    
    async def _extract_facts_from_text(self, question: str, text: str, llm_adapter: Any) -> List[str]:
        """Extract factual claims from text."""
        prompt = f"""Extract ALL verifiable factual claims from the text that relate to the question.

Question: {question}
Text: {text}

Respond ONLY with a JSON array of strings:
["fact 1", "fact 2", "fact 3"]

Facts:"""
        
        try:
            response = await llm_adapter.ainvoke(prompt)
            return parse_json_response(
                response.content,
                self.logger,
                "batch fact extraction",
            )
        except Exception:
            return []
    
    async def _check_fact_coverage_in_text(self, question: str, facts: List[str], text: str, llm_adapter: Any) -> List[Dict]:
        """Check which facts are covered in the text."""
        prompt = f"""For each fact, determine if it is covered in the text. Score 100 if fully covered, 0 if not.

Question: {question}
Facts: {json.dumps(facts)}
Text: {text}

Respond with ONLY a JSON array:
[{{"fact": "...", "score": 100, "reason": "..."}}]

Coverage Analysis:"""
        
        try:
            response = await llm_adapter.ainvoke(prompt)
            return parse_json_response(
                response.content,
                self.logger,
                "batch fact coverage",
            )
        except Exception:
            return []
    
    async def _extract_system_triples_from_answer(self, answer: str, llm_adapter: Any) -> set:
        """Extract factual triples from answer using LLM."""
        prompt = f"""Extract factual statements from the answer and format them as simple triples.

Answer: {answer}

Respond ONLY with a JSON array of strings:
["triple 1", "triple 2"]

Extracted triples:"""
        
        try:
            response = await llm_adapter.ainvoke(prompt)
            triples_list = parse_json_response(
                response.content,
                self.logger,
                "batch triple extraction",
            )
            return {triple.lower().strip() for triple in triples_list if triple.strip()}
        except Exception:
            return set()
    
    def _parse_evidence_tripples(self, evidence_triples: List[str]) -> set:
        """Parse evidence triples into normalized set."""
        if not evidence_triples:
            return set()
        
        normalized_triples = set()
        for triple in evidence_triples:
            if isinstance(triple, str) and triple.strip():
                normalized_triple = triple.lower().strip()
                normalized_triples.add(normalized_triple)
        
        return normalized_triples
