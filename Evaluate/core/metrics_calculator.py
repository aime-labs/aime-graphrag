#!/usr/bin/env python3
"""
Separated Metrics Calculator - focuses only on computing metrics from raw benchmark results.
This allows for efficient metrics computation with optimizations and caching.
"""

import asyncio
import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import logging
from collections import defaultdict

from config.evaluation_config import EvaluationConfig
from config.metric_config import MetricRegistry
from utils.logging_utils import EvaluationLogger
from utils.data_utils import DataProcessor
from adapters.model_manager import ModelManager
from adapters.llm_adapter import LLMAdapterFactory
from adapters.embedding_adapter import EmbeddingAdapterFactory
from core.adapter_pool import AdapterPool
from core.smart_cache import SmartCache
from metrics.text_metrics import TextMetrics
from metrics.semantic_metrics import SemanticMetrics
from metrics.retrieval_metrics import RetrievalMetrics
from metrics.triple_metrics import TripleMetrics
from metrics.hallucination_metrics import HallucinationMetrics
from metrics.ragas_metrics import RagasMetrics
from core.judge_tracker import JudgeTracker, TrackedLLMAdapter
import numpy as np


class MetricsCalculator:
    """Dedicated metrics calculator that processes raw benchmark results efficiently."""
    
    def __init__(self, config: EvaluationConfig, logger: EvaluationLogger):
        self.config = config
        self.logger = logger
        self.data_processor = DataProcessor(logger.logger)
        self.model_manager = ModelManager(logger.logger)
        
        # Initialize judge tracker for LLM interaction logging
        self.judge_tracker = JudgeTracker(logger.logger)
        self.metric_registry = MetricRegistry()
        
        # Initialize adapter pool for metrics computation
        pool_size = max(config.max_concurrent_tasks // 2, 5)
        self.adapter_pool = AdapterPool(logger.logger, max_pool_size=pool_size)
        
        # Initialize smart cache with metrics-specific optimizations
        cache_size = max(config.max_concurrent_tasks * 50, 1000)
        self.smart_cache = SmartCache(logger.logger, max_cache_size=cache_size)
        
        # Optimized semaphore for metrics computation
        self.semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        
        # Initialize metric components
        self.text_metrics = TextMetrics(logger.logger)
        self.semantic_metrics = SemanticMetrics(logger.logger)
        self.retrieval_metrics = RetrievalMetrics(logger.logger)
        self.triple_metrics = TripleMetrics(logger.logger)
        self.hallucination_metrics = HallucinationMetrics(logger.logger)
        self.ragas_metrics = RagasMetrics(logger.logger)
        
        # Model adapters
        self.judge_llm_adapter = None
        self.judge_llm_instance = None
        self.embeddings_adapter = None
        self.embeddings_instance = None
        self.graphrag_config = None
        
        # Results storage
        self.computed_metrics: List[Dict[str, Any]] = []
        self.metrics_cache_usage = {'hits': 0, 'misses': 0}
        
        # Checkpointing support with enhanced granular tracking
        self.checkpoint_file = Path(config.output_dir) / "metrics_checkpoint.json"
        self.checkpoint_interval = 10  # Save checkpoint every N tasks
        # Track completed (result_id, method) pairs to handle multiple methods per result
        self.processed_result_keys: Set[tuple] = set()  # Set of (result_id, method) tuples
        # Enhanced checkpoint metadata for better tracking and debugging
        self.checkpoint_metadata: Dict[tuple, Dict[str, Any]] = {}  # Maps (result_id, method) to metadata
    
    async def initialize(self):
        """Initialize the metrics calculator with models."""
        self.logger.logger.info("Initializing metrics calculator...")
        
        # Load GraphRAG configuration
        if self.config.config_path:
            self.graphrag_config = self.model_manager.load_config(
                self.config.config_path, 
                self.config.project_path
            )
        
        # Initialize models for metrics computation
        if self.graphrag_config:
            self.embeddings_instance = self.model_manager.get_embedding_model()
            self.embeddings_adapter = EmbeddingAdapterFactory.create_adapter(self.embeddings_instance, self.logger.logger)
            
            # Initialize judge LLM adapter
            judge_model_name = self.config.judge_llm_model or self.config.llm_model
            judge_api_base = self.config.judge_api_base_url or self.config.api_base_url
            judge_api_key = self.config.judge_api_key or self.config.api_key
            
            try:
                self.judge_llm_instance = self.model_manager.get_judge_chat_model(
                    judge_model_name, judge_api_base, judge_api_key, self.config.project_path
                )
                if self.judge_llm_instance:
                    # Prepare default LLM parameters for reproducible benchmarking
                    default_llm_params = {
                        'temperature': self.config.llm_temperature,
                        'max_tokens': self.config.llm_max_tokens,
                        'top_p': self.config.llm_top_p,
                        'top_k': self.config.llm_top_k,
                        'api_timeout': self.config.llm_api_timeout,
                    }
                    default_judge_params = default_llm_params.copy()
                    if self.config.judge_llm_max_tokens is not None:
                        default_judge_params['max_tokens'] = self.config.judge_llm_max_tokens
                    if self.config.judge_llm_api_timeout is not None:
                        default_judge_params['api_timeout'] = self.config.judge_llm_api_timeout
                    self.judge_llm_adapter = LLMAdapterFactory.create_adapter(
                        self.judge_llm_instance, self.logger.logger, default_params=default_judge_params
                    )
                    self.logger.logger.info(f"Initialized judge LLM for metrics: {judge_model_name}")
                else:
                    raise Exception("Could not initialize judge LLM")
            except Exception as e:
                self.logger.logger.error(f"Failed to initialize judge LLM for metrics: {str(e)}")
                raise
        
        self.logger.logger.info("Metrics calculator initialization complete")
    
    async def compute_metrics_from_benchmark(self, benchmark_results_path: str, 
                                           selected_metrics: Optional[List[str]] = None,
                                           resume_from_checkpoint: bool = False) -> str:
        """Compute metrics from raw benchmark results and return path to metrics file."""
        self.logger.logger.info(f"Starting metrics computation from {benchmark_results_path}")
        
        # Load checkpoint if resuming
        if resume_from_checkpoint and self.checkpoint_file.exists():
            self._load_checkpoint()
            self.logger.logger.info(f"Resuming from checkpoint with {len(self.processed_result_keys)} completed tasks")
        
        # Load benchmark results
        benchmark_results = self._load_benchmark_results(benchmark_results_path)
        
        # Group results for optimized processing
        grouped_results = self._group_results_for_metrics(benchmark_results)
        
        self.logger.logger.info(f"Processing metrics for {len(benchmark_results)} results")
        
        # Create metrics computation tasks
        all_tasks = []
        task_info = []
        
        for result_group in grouped_results:
            for result in result_group['results']:
                # Skip if already processed (checkpoint resume)
                result_id = result.get('id', f"result_{len(all_tasks)}")
                method = result['method']
                result_key = (result_id, method)
                
                if result_key in self.processed_result_keys:
                    self.logger.logger.debug(f"Skipping already processed result: {result_id} ({method})")
                    continue
                
                # Get metrics required for this result type from registry
                architecture = self._get_architecture_from_method(result['method'])
                required_metrics = self.metric_registry.get_metrics_for_config(
                    result['question_type'], architecture
                )
                
                # If specific metrics were selected, filter to only those; otherwise use all required metrics
                if selected_metrics:
                    metrics_for_result = [m for m in required_metrics if m in selected_metrics]
                else:
                    metrics_for_result = required_metrics
                
                if metrics_for_result:
                    task = self._compute_metrics_for_result(result, metrics_for_result)
                    all_tasks.append(task)
                    task_info.append((result['question_type'], result['method']))
        
        # Execute metrics computation with optimized batching
        total_tasks = len(all_tasks)
        completed = 0
        
        # Optimize batch size for metrics computation
        batch_size = min(self.config.batch_size * 3, self.config.max_concurrent_tasks)
        
        self.logger.logger.info(f"Processing {total_tasks} metrics computation tasks in batches of {batch_size}")
        
        try:
            for batch_start in range(0, total_tasks, batch_size):
                batch_end = min(batch_start + batch_size, total_tasks)
                batch_tasks = all_tasks[batch_start:batch_end]
                
                try:
                    # Execute batch concurrently
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Process results
                    for i, result in enumerate(batch_results):
                        task_idx = batch_start + i
                        if isinstance(result, Exception):
                            # Enhanced error logging with context
                            q_type, method = task_info[task_idx] if task_idx < len(task_info) else ('unknown', 'unknown')
                            self.logger.logger.error(f"Metrics computation failed for task {task_idx}:")
                            self.logger.logger.error(f"  Question Type: {q_type}")
                            self.logger.logger.error(f"  Method: {method}")
                            self.logger.logger.error(f"  Error: {str(result)}")
                            # Try to get original result data for more context
                            if task_idx < len(all_tasks):
                                try:
                                    original_idx = task_idx
                                    result_counter = 0
                                    for result_group in grouped_results:
                                        for orig_result in result_group['results']:
                                            if result_counter == original_idx:
                                                self.logger.logger.error(f"  Question: {orig_result.get('question', 'N/A')[:200]}...")
                                                self.logger.logger.error(f"  Answer: {orig_result.get('final_answer', 'N/A')[:200]}...")
                                                self.logger.logger.error(f"  Context count: {len(orig_result.get('contexts', []))}")
                                                break
                                            result_counter += 1
                                except Exception as log_err:
                                    self.logger.logger.debug(f"Could not log detailed context: {log_err}")
                            error_metrics = self._create_error_metrics_result(task_idx, q_type, method, str(result))
                            self.computed_metrics.append(error_metrics)
                        elif isinstance(result, dict):
                            self.computed_metrics.append(result)
                            # Track processed (result_id, method) for checkpoint with enhanced metadata
                            result_id = result.get('result_id', f"result_{task_idx}")
                            method = result.get('method', 'unknown')
                            result_key = (result_id, method)
                            self.processed_result_keys.add(result_key)
                            # Store enhanced checkpoint metadata
                            self.checkpoint_metadata[result_key] = {
                                'question_type': result.get('question_type', 'unknown'),
                                'source': result.get('source', 'unknown'),
                                'question_id': result.get('question_id', result_id),
                                'timestamp': result.get('timestamp', datetime.now(timezone.utc).isoformat()),
                                'computation_time': result.get('computation_time', 0.0)
                            }
                    
                    # Update completed count
                    completed += len(batch_tasks)
                    
                    # Log progress
                    progress_pct = (completed / total_tasks) * 100
                    unique_results = len(set(m['result_id'] for m in self.computed_metrics))
                    self.logger.logger.info(
                        f"Metrics Progress: {completed}/{total_tasks} ({progress_pct:.1f}%) "
                        f"[{unique_results} unique results, {len(self.computed_metrics)} metric entries]"
                    )
                    
                    # Save intermediate metrics and checkpoint
                    if completed % batch_size == 0:
                        self._save_intermediate_metrics()
                    
                    # Save checkpoint at intervals
                    if completed % self.checkpoint_interval == 0:
                        self._save_checkpoint()
                    
                except Exception as e:
                    self.logger.logger.error(f"Metrics batch execution failed: {str(e)}")
                    # Create error results for failed batch
                    for i in range(len(batch_tasks)):
                        task_idx = batch_start + i
                        q_type, method = task_info[task_idx] if task_idx < len(task_info) else ('unknown', 'unknown')
                        error_metrics = self._create_error_metrics_result(task_idx, q_type, method, str(e))
                        self.computed_metrics.append(error_metrics)
                    completed += len(batch_tasks)
                    
                    # Save intermediate metrics and checkpoint even after batch failure
                    self._save_intermediate_metrics()
                    self._save_checkpoint()
                    
        except KeyboardInterrupt:
            self.logger.logger.info(f"Metrics computation interrupted by user after {completed}/{total_tasks} tasks")
            # Save all intermediate results and checkpoint before re-raising
            self._save_intermediate_metrics()
            self._save_checkpoint()
            self.logger.logger.info(f"Progress saved to checkpoint: {self.checkpoint_file}")
            self.logger.logger.info(f"To resume, run with --resume_checkpoint flag")
            raise
        
        # Final save
        metrics_results_path = self._save_final_metrics()
        
        # Clear checkpoint after successful completion
        self.clear_checkpoint()
        
        # Log metrics computation summary
        self._log_metrics_summary()
        
        self.logger.logger.info("Metrics computation complete")
        return metrics_results_path
    
    async def _compute_metrics_for_result(self, result: Dict[str, Any], metrics: List[str]) -> Dict[str, Any]:
        """Compute specified metrics for a single result with optimizations."""
        async with self.semaphore:
            start_time = time.time()
            
            try:
                # Extract data from result for metric computation
                # Keep ground_truth mapped to 'answer' for metric functions
                question_data = {
                    'id': result.get('id'),
                    'question': result.get('question', ''),
                    'question_type': result.get('question_type', 'Uncategorized'),
                    'source': result.get('source', 'unknown'),
                    'answer': result.get('ground_truth_answer', ''),  # Ground truth for metric computation
                    'evidence': result.get('evidence', []),
                    'evidence_tripples': result.get('evidence_tripples', [])
                }
                
                answer = result.get('final_answer', '')
                contexts = result.get('contexts', [])
                method = result.get('method', '')
                
                # Compute metrics with caching and optimization
                computed_metrics = await self._compute_metrics_optimized(
                    question_data, answer, contexts, metrics, method
                )
                
                # Create metrics result - pass question_data which now has all required fields
                metrics_result = self.data_processor.create_metric_result(
                    question_data, method, computed_metrics
                )
                
                # Add timing information
                metrics_result['computation_time'] = time.time() - start_time
                metrics_result['result_id'] = result.get('id', 'unknown')
                
                return metrics_result
                
            except Exception as e:
                # Return error result
                return self._create_error_metrics_result(
                    result.get('id', 'unknown'), 
                    result.get('question_type', 'unknown'),
                    result.get('method', 'unknown'),
                    str(e)
                )
    
    async def _compute_metrics_optimized(self, question: Dict[str, Any], answer: str, 
                                       contexts: List[str], metrics: List[str], method: str) -> Dict[str, float]:
        """Compute specified metrics with advanced optimizations and caching.
        
        Uses parallel execution like unified mode for better performance.
        """
        computed_metrics = {}
        
        # Create cache key for this computation
        cache_key = self._create_metrics_cache_key(question, answer, contexts, metrics, method)
        
        # Check if entire computation is cached
        cached_result = await self.smart_cache.get_metric_result("full_computation", {"cache_key": cache_key})
        if cached_result is not None:
            self.metrics_cache_usage['hits'] += 1
            return cached_result
        
        self.metrics_cache_usage['misses'] += 1
        
        # Log what metrics we're computing
        self.logger.logger.debug(f"Computing metrics: {metrics}")
        
        ground_truth = question.get('answer', '')
        question_text = question.get('question', '')
        question_type = question.get('question_type', 'Uncategorized')
        
        # Define compute function for each metric (similar to unified mode)
        async def compute_single_metric(metric: str) -> tuple:
            """Compute a single metric and return (metric_name, value)."""
            try:
                # Text-only metrics
                if metric == 'exact_match':
                    score = self.text_metrics.compute_exact_match_score(answer, ground_truth)
                    return metric, score
                elif metric == 'f1_score':
                    score = self.text_metrics.compute_f1_score(answer, ground_truth)
                    return metric, score
                elif metric == 'bert_score_f1':
                    loop = asyncio.get_event_loop()
                    score = await loop.run_in_executor(
                        None, self.text_metrics.compute_bert_score_f1, answer, ground_truth
                    )
                    return metric, score
                
                # Embedding-based metrics
                elif metric == 'semantic_similarity_percentage':
                    if self.embeddings_instance:
                        async with self.adapter_pool.get_embedding_adapter_context(
                            self.embeddings_instance, "main", self.smart_cache
                        ) as embed_adapter:
                            score = await self.semantic_metrics.compute_semantic_similarity_percentage(
                                answer, ground_truth, embed_adapter
                            )
                            return metric, score
                    return metric, 0.0
                
                # RAGAS metrics
                elif metric == 'ragas_faithfulness':
                    async with self.adapter_pool.get_llm_adapter_context(
                        self.judge_llm_instance, "judge", self.smart_cache
                    ) as judge_adapter:
                        # Wrap with tracking for judge logging
                        tracked_adapter = TrackedLLMAdapter(
                            llm_adapter=judge_adapter,
                            tracker=self.judge_tracker,
                            question_id=question.get('id', 'unknown'),
                            question=question_text,
                            answer=answer,
                            ground_truth=ground_truth,
                            contexts=contexts,
                            default_metric='ragas_faithfulness'
                        )
                        result = await self.ragas_metrics.compute_faithfulness(
                            question_text, answer, contexts, tracked_adapter
                        )
                        if result is not None:
                            score, details = result
                            return metric, score
                        return metric, 0.0
                
                elif metric == 'ragas_context_precision':
                    async with self.adapter_pool.get_llm_adapter_context(
                        self.judge_llm_instance, "judge", self.smart_cache
                    ) as judge_adapter:
                        tracked_adapter = TrackedLLMAdapter(
                            llm_adapter=judge_adapter,
                            tracker=self.judge_tracker,
                            question_id=question.get('id', 'unknown'),
                            question=question_text,
                            answer=answer,
                            ground_truth=ground_truth,
                            contexts=contexts,
                            default_metric='ragas_context_precision'
                        )
                        result = await self.ragas_metrics.compute_context_precision(
                            question_text, contexts, ground_truth, tracked_adapter
                        )
                        if result is not None:
                            score, details = result
                            return metric, score
                        return metric, 0.0
                
                elif metric == 'ragas_context_recall':
                    if not ground_truth:
                        return metric, np.nan
                    async with self.adapter_pool.get_llm_adapter_context(
                        self.judge_llm_instance, "judge", self.smart_cache
                    ) as judge_adapter:
                        tracked_adapter = TrackedLLMAdapter(
                            llm_adapter=judge_adapter,
                            tracker=self.judge_tracker,
                            question_id=question.get('id', 'unknown'),
                            question=question_text,
                            answer=answer,
                            ground_truth=ground_truth,
                            contexts=contexts,
                            default_metric='ragas_context_recall'
                        )
                        result = await self.ragas_metrics.compute_context_recall(
                            question_text, contexts, ground_truth, tracked_adapter
                        )
                        if result is not None:
                            score, details = result
                            return metric, score
                        return metric, 0.0
                
                elif metric == 'ragas_answer_relevance':
                    if self.embeddings_instance and self.judge_llm_instance:
                        async with self.adapter_pool.get_llm_adapter_context(
                            self.judge_llm_instance, "judge", self.smart_cache
                        ) as judge_adapter:
                            tracked_adapter = TrackedLLMAdapter(
                                llm_adapter=judge_adapter,
                                tracker=self.judge_tracker,
                                question_id=question.get('id', 'unknown'),
                                question=question_text,
                                answer=answer,
                                ground_truth=ground_truth,
                                contexts=contexts,
                                default_metric='ragas_answer_relevance'
                            )
                            async with self.adapter_pool.get_embedding_adapter_context(
                                self.embeddings_instance, "main", self.smart_cache
                            ) as embed_adapter:
                                result = await self.ragas_metrics.compute_answer_relevance(
                                    question_text, answer, tracked_adapter, embed_adapter
                                )
                                if result is not None:
                                    score, details = result
                                    return metric, score
                    return metric, 0.0
                
                elif metric == 'ragas_score':
                    # Computed after other RAGAS metrics
                    return metric, None
                
                # LLM-based metrics
                elif metric == 'factual_accuracy_grade':
                    async with self.adapter_pool.get_llm_adapter_context(
                        self.judge_llm_instance, "judge", self.smart_cache
                    ) as judge_adapter:
                        tracked_adapter = TrackedLLMAdapter(
                            llm_adapter=judge_adapter,
                            tracker=self.judge_tracker,
                            question_id=question.get('id', 'unknown'),
                            question=question_text,
                            answer=answer,
                            ground_truth=ground_truth,
                            contexts=contexts,
                            default_metric='factual_accuracy_grade'
                        )
                        class SharedResultsStub:
                            def __init__(self, q_type):
                                self.question_type = q_type
                                self.factual_accuracy_evaluation = None
                        shared_results = SharedResultsStub(question_type)
                        score = await self.semantic_metrics.compute_factual_accuracy_grade(
                            question_text, answer, ground_truth, tracked_adapter, shared_results
                        )
                        return metric, score
                
                # Classification metrics - grouped together
                elif metric in ['correct_answers_count', 'wrong_answers_count', 'dont_know_answers_count']:
                    # These will be computed together, return placeholder
                    return metric, 'COMPUTE_CLASSIFICATION'
                
                else:
                    self.logger.logger.warning(f"Unknown metric '{metric}', skipping")
                    return metric, 0.0
                    
            except Exception as e:
                self.logger.logger.warning(f"Metric {metric} failed: {str(e)}")
                return metric, 0.0
        
        # Separate classification metrics from others (they need to be computed together)
        classification_metrics = [m for m in metrics if m in ['correct_answers_count', 'wrong_answers_count', 'dont_know_answers_count']]
        independent_metrics = [m for m in metrics if m not in classification_metrics]
        
        # Execute independent metrics in parallel
        if independent_metrics:
            tasks = [compute_single_metric(metric) for metric in independent_metrics]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    self.logger.logger.error(f"Metric computation failed: {str(result)}")
                elif isinstance(result, tuple) and len(result) == 2:
                    metric_name, value = result
                    computed_metrics[metric_name] = value
        
        # Compute composite ragas_score if requested and individual metrics are available
        if 'ragas_score' in computed_metrics and computed_metrics.get('ragas_score') is None:
            computed_metrics['ragas_score'] = self._compute_composite_ragas_score(computed_metrics)
        
        # Compute classification metrics together
        if classification_metrics and self.judge_llm_instance:
            async with self.adapter_pool.get_llm_adapter_context(
                self.judge_llm_instance, "judge", self.smart_cache
            ) as judge_adapter:
                tracked_adapter = TrackedLLMAdapter(
                    llm_adapter=judge_adapter,
                    tracker=self.judge_tracker,
                    question_id=question.get('id', 'unknown'),
                    question=question_text,
                    answer=answer,
                    ground_truth=ground_truth,
                    contexts=contexts,
                    default_metric='answer_classification'
                )
                try:
                    classification_results = await self.semantic_metrics.compute_answer_classification(
                        question_text, answer, ground_truth, tracked_adapter, question_type=question_type
                    )
                    for metric in classification_metrics:
                        if metric in classification_results:
                            computed_metrics[metric] = classification_results[metric]
                        else:
                            computed_metrics[metric] = 0
                except Exception as e:
                    self.logger.logger.warning(f"Classification metrics failed: {str(e)}")
                    for metric in classification_metrics:
                        computed_metrics[metric] = 0
        
        # Cache the entire computation result
        await self.smart_cache.cache_metric_result("full_computation", {"cache_key": cache_key}, computed_metrics)
        
        self.logger.logger.debug(f"Computed metrics result: {list(computed_metrics.keys())}")
        return computed_metrics
    
    def _compute_composite_ragas_score(self, metrics: Dict[str, float]) -> float:
        """Compute composite RAGAS score from individual metrics."""
        ragas_components = ['ragas_faithfulness', 'ragas_context_precision', 'ragas_context_recall', 'ragas_answer_relevance']
        scores = []
        for component in ragas_components:
            if component in metrics and metrics[component] is not None and not np.isnan(metrics[component]):
                scores.append(metrics[component])
        
        if scores:
            return sum(scores) / len(scores)
        return 0.0
    
    async def _compute_text_metrics_batch(self, question: Dict[str, Any], answer: str, 
                                        metrics: List[str]) -> Dict[str, float]:
        """Compute text-only metrics in batch for efficiency."""
        results = {}
        ground_truth = question.get('answer', '')
        
        for metric in metrics:
            try:
                if metric == 'exact_match':
                    results[metric] = self.text_metrics.compute_exact_match_score(answer, ground_truth)
                elif metric == 'f1_score':
                    results[metric] = self.text_metrics.compute_f1_score(answer, ground_truth)
                elif metric == 'bert_score_f1':
                    # BERTScore is CPU/GPU intensive, run in executor
                    loop = asyncio.get_event_loop()
                    score = await loop.run_in_executor(
                        None,
                        self.text_metrics.compute_bert_score_f1,
                        answer,
                        ground_truth
                    )
                    results[metric] = score
            except Exception as e:
                self.logger.logger.warning(f"Text metric {metric} failed: {str(e)}")
                results[metric] = 0.0
        
        return results
    
    async def _compute_llm_metrics_batch(self, question: Dict[str, Any], answer: str, 
                                       contexts: List[str], metrics: List[str], 
                                       judge_adapter: Any, question_type: str = 'Uncategorized') -> Dict[str, float]:
        """Compute LLM-based metrics in batch for efficiency."""
        results = {}
        ground_truth = question.get('answer', '')
        question_text = question.get('question', '')
        
        # Handle factual accuracy grade
        if 'factual_accuracy_grade' in metrics:
            try:
                # Create a shared_results stub for the penalty system
                class SharedResultsStub:
                    def __init__(self, q_type):
                        self.question_type = q_type
                        self.factual_accuracy_evaluation = None
                
                shared_results = SharedResultsStub(question_type)
                score = await self.semantic_metrics.compute_factual_accuracy_grade(
                    question_text, answer, ground_truth, judge_adapter, shared_results
                )
                results['factual_accuracy_grade'] = score
            except Exception as e:
                self.logger.logger.warning(f"Factual accuracy grade failed: {str(e)}")
                results['factual_accuracy_grade'] = "N/A"
        
        # Handle context relevance
        if 'context_relevance' in metrics:
            try:
                score = await self.retrieval_metrics.compute_context_relevance(
                    question_text, contexts, judge_adapter
                )
                results['context_relevance'] = score
            except Exception as e:
                self.logger.logger.warning(f"Context relevance failed: {str(e)}")
                results['context_relevance'] = 0.0
        
        # Handle triple metrics
        triple_metric_names = ['triple_em', 'triple_f1', 'triple_precision', 'triple_recall']
        requested_triple_metrics = [m for m in metrics if m in triple_metric_names]
        
        if requested_triple_metrics:
            try:
                evidence_tripples = question.get('evidence_tripples', [])
                if evidence_tripples:
                    triple_metrics = await self.triple_metrics.compute_triple_metrics_improved(
                        question_text, answer, contexts, evidence_tripples, judge_adapter
                    )
                    for metric in requested_triple_metrics:
                        results[metric] = triple_metrics.get(metric, 0.0)
                else:
                    for metric in requested_triple_metrics:
                        results[metric] = 0.0
            except Exception as e:
                self.logger.logger.warning(f"Triple metrics failed: {str(e)}")
                for metric in requested_triple_metrics:
                    results[metric] = 0.0
        
        return results
    
    def _load_benchmark_results(self, benchmark_results_path: str) -> List[Dict[str, Any]]:
        """Load benchmark results from file."""
        try:
            with open(benchmark_results_path, 'r') as f:
                results = json.load(f)
            self.logger.logger.info(f"Loaded {len(results)} benchmark results from {benchmark_results_path}")
            return results
        except Exception as e:
            self.logger.logger.error(f"Failed to load benchmark results: {str(e)}")
            raise
    
    def _group_results_for_metrics(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group results by similar characteristics for optimized processing."""
        # Group by method and question type for cache efficiency
        grouped = defaultdict(list)
        
        for result in results:
            key = (result.get('method', 'unknown'), result.get('question_type', 'unknown'))
            grouped[key].append(result)
        
        return [{'key': key, 'results': group_results} for key, group_results in grouped.items()]
    
    def _get_architecture_from_method(self, method: str) -> str:
        """Get architecture type from method name."""
        if method in ['local_search', 'global_search']:
            return 'graphrag'
        elif method == 'basic_search':
            return 'rag'
        elif method == 'llm_with_context':
            return 'direct_llm'
        else:
            return 'unknown'
    
    def _create_metrics_cache_key(self, question: Dict[str, Any], answer: str, 
                                contexts: List[str], metrics: List[str], method: str) -> str:
        """Create cache key for metrics computation."""
        key_data = {
            'question': question.get('question', ''),
            'answer': answer,
            'contexts_hash': hash(tuple(contexts)) if contexts else 0,
            'metrics': sorted(metrics),
            'method': method
        }
        return json.dumps(key_data, sort_keys=True)
    
    def _create_error_metrics_result(self, result_id: str, q_type: str, method: str, error_msg: str) -> Dict[str, Any]:
        """Create error metrics result with detailed logging."""
        error_result = {
            'result_id': result_id,
            'question_type': q_type,
            'method': method,
            'metrics': {},
            'error_message': error_msg,
            'computation_time': 0.0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        # Log the error creation for tracking
        self.logger.logger.error(f"Created error metrics result for {result_id}: {error_msg[:100]}")
        return error_result
    
    def _save_intermediate_metrics(self):
        """Save intermediate metrics results."""
        try:
            output_path = Path(self.config.output_dir) / "metrics_computed.json"
            with open(output_path, 'w') as f:
                json.dump(self.computed_metrics, f, indent=2, default=str)
            self.logger.logger.debug(f"Intermediate metrics saved to {output_path}")
        except Exception as e:
            self.logger.logger.warning(f"Failed to save intermediate metrics: {str(e)}")
    
    def _save_final_metrics(self) -> str:
        """Save final metrics and return path."""
        try:
            output_path = Path(self.config.output_dir) / "metrics_computed.json"
            with open(output_path, 'w') as f:
                json.dump(self.computed_metrics, f, indent=2, default=str)
            
            # Also save a summary
            summary_path = Path(self.config.output_dir) / "metrics_summary.json"
            summary = {
                'total_metrics_computed': len(self.computed_metrics),
                'cache_usage': self.metrics_cache_usage,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.logger.info(f"Final metrics saved to {output_path}")
            self.logger.logger.info(f"Metrics summary saved to {summary_path}")
            
            # Save judge evaluations (if enabled in config)
            if self.config.enable_judge_logging:
                self._save_judge_evaluations()
            
            return str(output_path)
            
        except Exception as e:
            self.logger.logger.error(f"Failed to save final metrics: {str(e)}")
            raise
    
    def _save_judge_evaluations(self):
        """Save judge LLM interaction logs to judge.json."""
        try:
            output_path = Path(self.config.output_dir) / "judge.json"
            
            # Get all interactions from the tracker
            interactions = self.judge_tracker.get_all_interactions()
            
            # Get summary statistics
            summary = self.judge_tracker.get_summary()
            
            # Combine into output structure
            judge_data = {
                "summary": summary,
                "interactions": interactions
            }
            
            with open(output_path, 'w') as f:
                json.dump(judge_data, f, indent=2, default=str)
            
            self.logger.logger.info(f"Judge evaluations saved to {output_path} ({len(interactions)} interactions)")
            
        except Exception as e:
            self.logger.logger.warning(f"Failed to save judge evaluations: {str(e)}")
    
    def _log_metrics_summary(self):
        """Log metrics computation summary."""
        # Cache statistics
        cache_stats = self.smart_cache.get_cache_stats()
        self.logger.logger.info(f"Metrics Cache Statistics: {cache_stats}")
        
        # Pool statistics
        pool_stats = self.adapter_pool.get_pool_stats()
        self.logger.logger.info(f"Metrics Adapter Pool Statistics: {pool_stats}")
        
        # Metrics computation statistics
        total_computed = len(self.computed_metrics)
        error_computed = len([m for m in self.computed_metrics if m.get('error_message')])
        success_rate = ((total_computed - error_computed) / total_computed * 100) if total_computed > 0 else 0
        
        self.logger.logger.info(f"Metrics Computation Summary:")
        self.logger.logger.info(f"  Total metrics computed: {total_computed}")
        self.logger.logger.info(f"  Error computations: {error_computed}")
        self.logger.logger.info(f"  Success rate: {success_rate:.1f}%")
        self.logger.logger.info(f"  Cache usage: {self.metrics_cache_usage}")
    
    def get_computed_metrics(self) -> List[Dict[str, Any]]:
        """Get all computed metrics."""
        return self.computed_metrics.copy()

    def _save_checkpoint(self):
        """Save checkpoint with enhanced granular progress tracking."""
        try:
            # Derive processed keys from actual computed metrics for accuracy
            actual_processed_keys = [[m['result_id'], m['method']] for m in self.computed_metrics]
            
            # Build enhanced metadata structure for better tracking
            metadata_list = []
            for m in self.computed_metrics:
                result_key = (m['result_id'], m['method'])
                meta = {
                    'result_id': m['result_id'],
                    'method': m['method'],
                    'question_type': m.get('question_type', 'unknown'),
                    'source': m.get('source', 'unknown'),
                    'question_id': m.get('question_id', m['result_id']),
                    'timestamp': m.get('timestamp', datetime.now(timezone.utc).isoformat()),
                    'computation_time': m.get('computation_time', 0.0),
                    'error_message': m.get('error_message', '')
                }
                metadata_list.append(meta)
            
            # Group by different dimensions for analysis
            by_question_type = defaultdict(list)
            by_method = defaultdict(list)
            by_source = defaultdict(list)
            
            for meta in metadata_list:
                by_question_type[meta['question_type']].append((meta['result_id'], meta['method']))
                by_method[meta['method']].append((meta['result_id'], meta['question_type']))
                by_source[meta['source']].append((meta['result_id'], meta['method']))
            
            checkpoint_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'processed_result_keys': actual_processed_keys,  # List of [result_id, method] pairs
                'total_computed': len(self.computed_metrics),
                'unique_result_ids': len(set(m['result_id'] for m in self.computed_metrics)),
                'cache_usage': self.metrics_cache_usage.copy(),
                # Enhanced granular tracking
                'metadata': metadata_list,
                'summary_by_question_type': {k: len(v) for k, v in by_question_type.items()},
                'summary_by_method': {k: len(v) for k, v in by_method.items()},
                'summary_by_source': {k: len(v) for k, v in by_source.items()},
                'error_count': len([m for m in metadata_list if m['error_message']])
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            self.logger.logger.debug(
                f"Checkpoint saved: {len(actual_processed_keys)} tasks processed "
                f"({checkpoint_data['unique_result_ids']} unique results, "
                f"{checkpoint_data['error_count']} errors)"
            )
        except Exception as e:
            self.logger.logger.warning(f"Failed to save checkpoint: {str(e)}")
    
    def _load_checkpoint(self):
        """Load checkpoint with enhanced granular tracking to resume computation."""
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Load processed keys - handle both old format (strings) and new format (tuples)
            processed_keys_data = checkpoint_data.get('processed_result_keys', 
                                                     checkpoint_data.get('processed_result_ids', []))
            
            # Convert to set of tuples
            if processed_keys_data and isinstance(processed_keys_data[0], list):
                # New format: list of [result_id, method] pairs
                self.processed_result_keys = set(tuple(pair) for pair in processed_keys_data)
            elif processed_keys_data and isinstance(processed_keys_data[0], str):
                # Old format: just result_ids - we'll need to derive method from computed metrics
                self.logger.logger.warning("Old checkpoint format detected, will derive methods from computed metrics")
                self.processed_result_keys = set()
            else:
                self.processed_result_keys = set()
            
            # Load enhanced metadata if available
            if 'metadata' in checkpoint_data:
                for meta in checkpoint_data['metadata']:
                    result_key = (meta['result_id'], meta['method'])
                    self.checkpoint_metadata[result_key] = meta
                self.logger.logger.info(f"Loaded metadata for {len(self.checkpoint_metadata)} processed tasks")
            
            self.metrics_cache_usage = checkpoint_data.get('cache_usage', {'hits': 0, 'misses': 0})
            
            # Try to load intermediate metrics if they exist
            intermediate_path = Path(self.config.output_dir) / "metrics_computed.json"
            if intermediate_path.exists():
                with open(intermediate_path, 'r') as f:
                    all_metrics = json.load(f)
                
                # Deduplicate metrics by (result_id, method), keeping latest timestamp
                metrics_by_key = defaultdict(list)
                for metric in all_metrics:
                    key = (metric['result_id'], metric['method'])
                    metrics_by_key[key].append(metric)
                
                # Keep only the latest entry for each (result_id, method)
                deduped_metrics = []
                duplicate_count = 0
                for key, metrics_list in metrics_by_key.items():
                    if len(metrics_list) > 1:
                        duplicate_count += len(metrics_list) - 1
                        # Sort by timestamp and keep the latest
                        metrics_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                    deduped_metrics.append(metrics_list[0])
                
                self.computed_metrics = deduped_metrics
                
                # Update processed_result_keys and metadata from actual computed metrics
                self.processed_result_keys = set(
                    (m['result_id'], m['method']) for m in self.computed_metrics
                )
                
                # Rebuild metadata from computed metrics if not loaded from checkpoint
                if not self.checkpoint_metadata:
                    for m in self.computed_metrics:
                        result_key = (m['result_id'], m['method'])
                        self.checkpoint_metadata[result_key] = {
                            'question_type': m.get('question_type', 'unknown'),
                            'source': m.get('source', 'unknown'),
                            'question_id': m.get('question_id', m['result_id']),
                            'timestamp': m.get('timestamp', 'unknown'),
                            'computation_time': m.get('computation_time', 0.0),
                            'error_message': m.get('error_message', '')
                        }
                
                if duplicate_count > 0:
                    self.logger.logger.warning(
                        f"Removed {duplicate_count} duplicate entries during checkpoint load"
                    )
                
                self.logger.logger.info(
                    f"Loaded {len(self.computed_metrics)} deduplicated computed metrics from checkpoint"
                )
            
            checkpoint_time = checkpoint_data.get('timestamp', 'unknown')
            self.logger.logger.info(f"Checkpoint loaded from {checkpoint_time}")
            
            # Log enhanced summary information
            if 'summary_by_question_type' in checkpoint_data:
                self.logger.logger.info(f"  By question type: {checkpoint_data['summary_by_question_type']}")
            if 'summary_by_method' in checkpoint_data:
                self.logger.logger.info(f"  By method: {checkpoint_data['summary_by_method']}")
            if 'error_count' in checkpoint_data:
                self.logger.logger.info(f"  Total errors: {checkpoint_data['error_count']}")
            
            self.logger.logger.info(
                f"Resuming with {len(self.processed_result_keys)} completed tasks "
                f"({len(set(k[0] for k in self.processed_result_keys))} unique results)"
            )
        except Exception as e:
            self.logger.logger.error(f"Failed to load checkpoint: {str(e)}")
            self.logger.logger.info("Starting fresh computation")
    
    def clear_checkpoint(self):
        """Clear checkpoint file after successful completion."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                self.logger.logger.info("Checkpoint cleared after successful completion")
        except Exception as e:
            self.logger.logger.warning(f"Failed to clear checkpoint: {str(e)}")

    async def cleanup(self):
        """Cleanup resources including adapters and sessions."""
        self.logger.logger.info("Cleaning up metrics calculator resources...")
        
        # Cleanup judge LLM adapter
        if self.judge_llm_adapter and hasattr(self.judge_llm_adapter, 'close'):
            try:
                await self.judge_llm_adapter.close()
            except Exception as e:
                self.logger.logger.warning(f"Error cleaning up judge LLM adapter: {e}")
        
        # Cleanup embeddings adapter (if it has a close method)
        if self.embeddings_adapter and hasattr(self.embeddings_adapter, 'close'):
            try:
                await self.embeddings_adapter.close()
            except Exception as e:
                self.logger.logger.warning(f"Error cleaning up embeddings adapter: {e}")
        
        # Cleanup adapter pool
        if hasattr(self, 'adapter_pool') and self.adapter_pool:
            try:
                await self.adapter_pool.cleanup()
            except Exception as e:
                self.logger.logger.warning(f"Error cleaning up adapter pool: {e}")
        
        # Cleanup smart cache
        if hasattr(self, 'smart_cache') and self.smart_cache:
            try:
                await self.smart_cache.clear_all_caches()
            except Exception as e:
                self.logger.logger.warning(f"Error cleaning up smart cache: {e}")
        
        self.logger.logger.info("Metrics calculator cleanup complete")
