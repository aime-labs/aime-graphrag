import asyncio
import json
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

from config.evaluation_config import EvaluationConfig
from config.metric_config import MetricRegistry
from utils.logging_utils import EvaluationLogger
from utils.data_utils import DataLoader, DataProcessor
from adapters.model_manager import ModelManager
from adapters.llm_adapter import LLMAdapterFactory
from adapters.embedding_adapter import EmbeddingAdapterFactory
from core.query_runner import QueryRunnerFactory
from core.result_processor import ResultProcessor
from core.adapter_pool import AdapterPool
from core.smart_cache import SmartCache, CachedLLMAdapter, CachedEmbeddingAdapter
from core.resource_monitor import ResourceMonitor, AdaptiveSemaphore
from core.judge_tracker import JudgeTracker, TrackedLLMAdapter
from core.context_builder import build_llm_context
from metrics.text_metrics import TextMetrics
from metrics.semantic_metrics import SemanticMetrics
from metrics.ragas_metrics import RagasMetrics
from metrics.pairwise_metrics import PairwiseMetrics
from collections import defaultdict


class Evaluator:
    """Main evaluation orchestrator."""
    
    def __init__(self, config: EvaluationConfig, logger: EvaluationLogger):
        self.config = config
        self.logger = logger
        self.data_loader = DataLoader(logger.logger)
        self.data_processor = DataProcessor(logger.logger)
        self.model_manager = ModelManager(logger.logger)
        self.metric_registry = MetricRegistry()
        self.result_processor = ResultProcessor(logger, self.data_processor)
        
        # Initialize adapter pool for better performance
        pool_size = max(config.max_concurrent_tasks // 2, 5)  # Reasonable pool size
        self.adapter_pool = AdapterPool(logger.logger, max_pool_size=pool_size)
        
        # Initialize smart cache for LLM responses and embeddings
        cache_size = max(config.max_concurrent_tasks * 50, 1000)  # Larger cache for high concurrency
        self.smart_cache = SmartCache(logger.logger, max_cache_size=cache_size)
        
        # Use simple semaphore instead of adaptive one for better performance
        self.semaphore = asyncio.Semaphore(config.max_concurrent_tasks * 2)  # Higher limit for API-bound tasks
        
        # Initialize metric components (only the required ones)
        self.text_metrics = TextMetrics(logger.logger)
        self.semantic_metrics = SemanticMetrics(logger.logger)
        self.ragas_metrics = RagasMetrics(logger.logger)
        self.pairwise_metrics = PairwiseMetrics(logger.logger)
        
        # Initialize judge tracker for tracking all judge LLM interactions
        self.judge_tracker = JudgeTracker(logger.logger)
        
        # Model adapters and instances
        self.llm_adapter = None
        self.judge_llm_adapter = None
        self.embeddings_adapter = None
        self.llm_instance = None
        self.judge_llm_instance = None
        self.embeddings_instance = None
        self.reranker = None  # Reranker for context precision
        self.graphrag_config = None
        self.index_files = {}
        self.novel_contexts = {}
    
    async def initialize(self):
        """Initialize the evaluator with models and data."""
        self.logger.logger.info("Initializing evaluator...")
        
        # Load GraphRAG configuration
        if self.config.config_path:
            self.graphrag_config = self.model_manager.load_config(
                self.config.config_path, 
                self.config.project_path
            )
        
        # Load index files
        if self.config.project_path:
            project_output_dir = Path(self.config.project_path) / 'output'
            self.index_files = self.model_manager.load_index_files(str(project_output_dir))
        
        # Initialize LLM and embeddings
        if self.graphrag_config:
            self.llm_instance = self.model_manager.get_chat_model()
            self.embeddings_instance = self.model_manager.get_embedding_model()
            
            # Prepare default LLM parameters for reproducible benchmarking
            default_llm_params = {
                'temperature': self.config.llm_temperature,
                'max_tokens': self.config.llm_max_tokens,
                'top_p': self.config.llm_top_p,
                'top_k': self.config.llm_top_k,
                'api_timeout': self.config.llm_api_timeout,
            }
            
            self.llm_adapter = LLMAdapterFactory.create_adapter(
                self.llm_instance, self.logger.logger, default_params=default_llm_params
            )
            self.embeddings_adapter = EmbeddingAdapterFactory.create_adapter(self.embeddings_instance, self.logger.logger)
            
            # Initialize separate judge LLM adapter
            judge_model_name = self.config.judge_llm_model or self.config.llm_model
            judge_api_base = self.config.judge_api_base_url or self.config.api_base_url
            judge_api_key = self.config.judge_api_key or self.config.api_key
            
            try:
                self.judge_llm_instance = self.model_manager.get_judge_chat_model(
                    judge_model_name, judge_api_base, judge_api_key, self.config.project_path
                )
                if self.judge_llm_instance:
                    default_judge_params = default_llm_params.copy()
                    if self.config.judge_llm_max_tokens is not None:
                        default_judge_params['max_tokens'] = self.config.judge_llm_max_tokens
                    if self.config.judge_llm_api_timeout is not None:
                        default_judge_params['api_timeout'] = self.config.judge_llm_api_timeout
                    self.judge_llm_adapter = LLMAdapterFactory.create_adapter(
                        self.judge_llm_instance, self.logger.logger, default_params=default_judge_params
                    )
                    self.logger.logger.info(f"Initialized judge LLM: {judge_model_name}")
                else:
                    # Fallback to main LLM adapter
                    self.judge_llm_adapter = self.llm_adapter
                    self.judge_llm_instance = self.llm_instance
                    self.logger.logger.warning("Using main LLM adapter for judge functionality")
            except Exception as e:
                self.logger.logger.warning(f"Failed to initialize judge LLM, using main LLM: {str(e)}")
                self.judge_llm_adapter = self.llm_adapter
                self.judge_llm_instance = self.llm_instance
        
        # Initialize reranker for context precision metric
        try:
            reranker_config = getattr(self.config, 'reranker', {})
            if isinstance(reranker_config, dict):
                reranker_enabled = reranker_config.get('enabled', True)
                reranker_type = reranker_config.get('type', 'bge')
                reranker_model = reranker_config.get('model_name', None)
                reranker_device = reranker_config.get('device', 'cuda')
                reranker_max_length = reranker_config.get('max_length', 512)
                
                if reranker_enabled and reranker_type != 'none':
                    self.reranker = self.model_manager.get_reranker(
                        reranker_type=reranker_type,
                        model_name=reranker_model,
                        device=reranker_device,
                        max_length=reranker_max_length
                    )
                    if self.reranker:
                        self.logger.logger.info(
                            f"Initialized reranker: {reranker_type} "
                            f"(model: {reranker_model or 'default'}, device: {reranker_device})"
                        )
                    else:
                        self.logger.logger.warning("Reranker initialization returned None")
                else:
                    self.logger.logger.info("Reranker disabled in configuration")
            else:
                self.logger.logger.debug("No reranker configuration found, context precision will use original ordering")
        except Exception as e:
            self.logger.logger.warning(f"Failed to initialize reranker: {e}. Context precision will use original ordering.")
            self.reranker = None
        
        # Load novel contexts asynchronously
        if self.config.input_json_path:
            self.novel_contexts = await self.data_loader.load_novel_contexts_async(self.config.input_json_path)
        
        self.logger.logger.info("Evaluator initialization complete")
    
    async def run_pairwise_comparison(self):
        """Run pairwise comparison between methods."""
        if not self.config.pairwise_evaluation:
            return
            
        self.logger.logger.info("Starting pairwise comparison...")
        
        # Get all results
        results = self.result_processor.results
        if not results:
            self.logger.logger.warning("No results to compare")
            return
            
        # Group results by question
        results_by_question = defaultdict(dict)
        for res in results:
            # Use question text as key
            q_text = res.get('question', '')
            method = res.get('method', '')
            if q_text and method:
                results_by_question[q_text][method] = res
        
        methods = self.config.methods
        if len(methods) < 2:
            self.logger.logger.warning("Need at least 2 methods for pairwise comparison")
            return
            
        # Compare every pair of methods
        comparisons = []
        
        # Generate pairs (A vs B)
        pairs = []
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                pairs.append((methods[i], methods[j]))
        
        self.logger.logger.info(f"Comparing {len(pairs)} pairs of methods for {len(results_by_question)} questions")
        
        for q_text, method_results in results_by_question.items():
            # Get ground truth from one of the results (assuming it's consistent)
            first_res = next(iter(method_results.values()))
            ground_truth = first_res.get('ground_truth', '')
            
            for method_a, method_b in pairs:
                if method_a in method_results and method_b in method_results:
                    res_a = method_results[method_a]
                    res_b = method_results[method_b]
                    
                    ans_a = res_a.get('processed_answer', '')
                    ans_b = res_b.get('processed_answer', '')
                    
                    # Skip if answers are empty or errors
                    if not ans_a or not ans_b or "[ERROR" in ans_a or "[ERROR" in ans_b:
                        continue
                        
                    comparison = await self.pairwise_metrics.compare_answers(
                        question=q_text,
                        answer_a=ans_a,
                        answer_b=ans_b,
                        ground_truth=ground_truth,
                        llm_adapter=self.judge_llm_adapter
                    )
                    
                    comparisons.append({
                        "question": q_text,
                        "method_a": method_a,
                        "method_b": method_b,
                        "result": comparison
                    })
        
        # Save comparisons
        output_file = Path(self.config.output_dir) / "pairwise_results.json"
        with open(output_file, "w") as f:
            json.dump(comparisons, f, indent=2)
            
        self.logger.logger.info(f"Pairwise comparison complete. Saved {len(comparisons)} comparisons to {output_file}")

    def _get_enriched_context_for_llm(self, source: str) -> str:
        """Get context for LLM evaluation.
        
        Uses the shared build_llm_context function to ensure consistency between
        unified (evaluator) mode and benchmark mode.
        
        The llm_with_context method tests LLM comprehension using ALL documents
        from novel.json/medical.json to provide the same context that GraphRAG has access to.
        This ensures a fair comparison between direct LLM and GraphRAG approaches.
        
        Large contexts are handled by DirectLLMRunner's chunk-based parallel processing.
        
        Args:
            source: The source identifier for the primary document (used for logging)
        """
        return build_llm_context(self.novel_contexts, source, self.logger.logger)
    
    async def evaluate(self, questions_path: str):
        """Run the complete evaluation pipeline."""
        self.logger.logger.info(f"Starting evaluation with {len(self.config.methods)} methods")
        
        # Load questions asynchronously with optional type filtering
        questions = await self.data_loader.load_questions_async(questions_path, self.config.question_types_filter)
        grouped_questions = self.data_loader.group_questions_by_type(questions)
        
        # Preload contexts for questions that need them
        if self.novel_contexts:
            await self.data_loader.preload_contexts_for_questions(questions, self.novel_contexts)
        
        # Apply sample limit
        if self.config.max_samples_per_type:
            for q_type in grouped_questions:
                grouped_questions[q_type] = grouped_questions[q_type][:self.config.max_samples_per_type]
        
        # Log configuration
        self.logger.log_configuration({
            'methods': self.config.methods,
            'question_types': list(grouped_questions.keys()),
            'total_questions': sum(len(qs) for qs in grouped_questions.values()),
            'max_samples_per_type': self.config.max_samples_per_type
        })
        
        # Create evaluation tasks with improved task distribution
        all_tasks = []
        task_info = []
        # CRITICAL-002 FIX: Store deep copies of original questions to preserve data in error handling
        # Use deep copy to prevent modifications during task execution from affecting stored copies
        from copy import deepcopy
        all_questions = []
        
        for q_type, questions_list in grouped_questions.items():
            for question in questions_list:
                for method in self.config.methods:
                    # Get appropriate metrics for this configuration
                    architecture = QueryRunnerFactory.get_method_architecture(method)
                    metrics = self.metric_registry.get_metrics_for_config(q_type, architecture)
                    
                    task = self._evaluate_single_question(
                        question, method, metrics, q_type
                    )
                    
                    all_tasks.append(task)
                    task_info.append((q_type, method))
                    # CRITICAL-002 FIX: Store deep copy to ensure original data is preserved
                    all_questions.append(deepcopy(question))

        # Execute tasks with optimized parallel processing
        total_tasks = len(all_tasks)
        completed = 0
        
        # Process tasks in optimized batches with better concurrency
        batch_size = min(self.config.batch_size * 5, self.config.max_concurrent_tasks)  # Larger batches
        
        self.logger.logger.info(f"Processing {total_tasks} tasks in batches of {batch_size}")
        
        try:
            for batch_start in range(0, total_tasks, batch_size):
                batch_end = min(batch_start + batch_size, total_tasks)
                batch_tasks = all_tasks[batch_start:batch_end]
                
                try:
                    # Execute batch concurrently
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Process results
                    for i, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            self.logger.logger.error(f"Task failed: {str(result)}")
                            task_idx = batch_start + i
                            # CRITICAL-002 FIX: Safely extract task info with bounds checking
                            q_type, method = task_info[task_idx] if task_idx < len(task_info) else ('Fact Retrieval', 'unknown')
                            # CRITICAL-002 FIX: Use stored deep copy to preserve original question data
                            original_question = all_questions[task_idx] if task_idx < len(all_questions) else None
                            if original_question:
                                # Use the original question directly (already has all fields)
                                error_result = self.data_processor.create_evaluation_result(
                                    original_question, method, f"[ERROR: {str(result)}]", [], {}, str(result),
                                    raw_answer=f"[ERROR: {str(result)}]", processed_answer=f"[ERROR: {str(result)}]"
                                )
                            else:
                                # Fallback for missing question data (should not happen)
                                self.logger.logger.error(f"No original question data for task {task_idx}")
                                error_question = {
                                    'id': f'error_{task_idx}',
                                    'question': 'Task failed - original question data missing',
                                    'question_type': q_type,
                                    'answer': None,
                                    'source': 'unknown',
                                    'evidence': [],
                                    'evidence_tripples': []
                                }
                                error_result = self.data_processor.create_evaluation_result(
                                    error_question, method, f"[ERROR: {str(result)}]", [], {}, str(result),
                                    raw_answer=f"[ERROR: {str(result)}]", processed_answer=f"[ERROR: {str(result)}]"
                                )
                            self.result_processor.add_result(error_result)
                        elif isinstance(result, dict):
                            self.result_processor.add_result(result)
                            # Log progress for successful results
                            q_type, method = task_info[batch_start + i] if batch_start + i < len(task_info) else ('unknown', 'unknown')
                            self.logger.log_evaluation_progress(completed + i + 1, total_tasks, q_type, method)
                        else:
                            self.logger.logger.warning(f"Unexpected result type: {type(result)}")
                    
                    # Update completed count
                    completed += len(batch_tasks)
                    
                    # Log progress
                    progress_pct = (completed / total_tasks) * 100
                    self.logger.logger.info(f"Progress: {completed}/{total_tasks} ({progress_pct:.1f}%)")
                    
                    # Save intermediate results more frequently to ensure data safety
                    if completed % batch_size == 0:  # Save after every batch
                        self._save_intermediate_results()
                    
                except Exception as e:
                    self.logger.logger.error(f"Batch execution failed: {str(e)}")
                    # CRITICAL-002 FIX & CRITICAL-007 FIX: Don't create error results for whole batch
                    # Instead, mark this batch for retry and log which tasks were affected
                    failed_task_indices = []
                    for i in range(len(batch_tasks)):
                        task_idx = batch_start + i
                        failed_task_indices.append(task_idx)
                        q_type, method = task_info[task_idx] if task_idx < len(task_info) else ('Fact Retrieval', 'unknown')
                        # CRITICAL-002 FIX: Use stored deep copy to preserve original question data
                        original_question = all_questions[task_idx] if task_idx < len(all_questions) else None
                        if original_question:
                            error_result = self.data_processor.create_evaluation_result(
                                original_question, method, f"[BATCH ERROR: {str(e)}]", [], {}, str(e),
                                raw_answer=f"[BATCH ERROR: {str(e)}]", processed_answer=f"[BATCH ERROR: {str(e)}]"
                            )
                        else:
                            # Fallback for missing question data
                            self.logger.logger.error(f"No original question data for task {task_idx}")
                            error_question = {
                                'id': f'batch_error_{task_idx}',
                                'question': 'Batch failed - original question data missing',
                                'question_type': q_type,
                                'answer': None,
                                'source': 'unknown',
                                'evidence': [],
                                'evidence_tripples': []
                            }
                            error_result = self.data_processor.create_evaluation_result(
                                error_question, method, f"[BATCH ERROR: {str(e)}]", [], {}, str(e),
                                raw_answer=f"[BATCH ERROR: {str(e)}]", processed_answer=f"[BATCH ERROR: {str(e)}]"
                            )
                        self.result_processor.add_result(error_result)
                    
                    # CRITICAL-007 FIX: Log failed task indices for potential retry
                    self.logger.logger.warning(f"Batch failed affecting tasks {failed_task_indices[0]} to {failed_task_indices[-1]}")
                    completed += len(batch_tasks)
                    
                    # Save intermediate results even after batch failure
                    self._save_intermediate_results()
                    
        except KeyboardInterrupt:
            self.logger.logger.info(f"Evaluation interrupted by user after {completed}/{total_tasks} tasks")
            # Save all intermediate results before re-raising
            self._save_intermediate_results()
            raise
        
        # Final save
        self._save_final_results()
        
        # Run pairwise comparison if enabled
        await self.run_pairwise_comparison()
        
        # Generate and log summary
        self._log_evaluation_summary()
        
        self.logger.logger.info("Evaluation complete")
    
    async def _evaluate_single_question(self, question: Dict[str, Any], method: str, 
                                      metrics: List[str], question_type: str) -> Dict[str, Any]:
        """Evaluate a single question with a specific method."""
        async with self.semaphore:
            start_time = time.time()
            
            try:
                # Create query runner
                runner = QueryRunnerFactory.create_runner(
                    method, self.graphrag_config, self.logger.logger,
                    index_files=self.index_files, llm_adapter=self.llm_adapter
                )
                
                # Get prompt template
                prompt_template = self._load_prompt_template(question_type)
                
                # Get context for direct LLM methods
                context = None
                if method == 'llm_with_context':
                    source = question.get('source')
                    if source:
                        # Include all documents from novel.json for fair comparison with GraphRAG
                        context = self._get_enriched_context_for_llm(source)
                
                # Run query
                raw_answer, processed_answer, contexts = await runner.run_query(
                    question['question'], context, prompt_template
                )
                
                # Determine final answer based on whether we have prompt template
                final_answer = processed_answer if prompt_template else raw_answer
                
                # Compute metrics (track computation time)
                metrics_start_time = time.time()
                computed_metrics = await self._compute_metrics(
                    question, final_answer, contexts, metrics, method
                )
                metrics_computation_time = time.time() - metrics_start_time
                
                # Generate embeddings for the answer and ground truth
                answer_embedding = None
                ground_truth_embedding = None
                try:
                    async with self.adapter_pool.get_embedding_adapter_context(self.embeddings_instance, "main", self.smart_cache) as embed_adapter:
                        # Generate embeddings in parallel
                        answer_embedding, ground_truth_embedding = await asyncio.gather(
                            embed_adapter.aembed_query(final_answer),
                            embed_adapter.aembed_query(question.get('answer', '')),
                            return_exceptions=True
                        )
                        
                        # Handle embedding generation errors
                        if isinstance(answer_embedding, Exception):
                            self.logger.logger.warning(f"Failed to generate answer embedding: {str(answer_embedding)}")
                            answer_embedding = None
                        if isinstance(ground_truth_embedding, Exception):
                            self.logger.logger.warning(f"Failed to generate ground truth embedding: {str(ground_truth_embedding)}")
                            ground_truth_embedding = None
                            
                except Exception as e:
                    self.logger.logger.warning(f"Failed to generate embeddings: {str(e)}")
                
                # Create result with both raw and processed answers
                # Note: Embeddings are generated for semantic similarity metrics but NOT stored in results.json
                # to keep the output file size manageable
                result = self.data_processor.create_evaluation_result(
                    question, method, final_answer, contexts, computed_metrics,
                    raw_answer=raw_answer, processed_answer=processed_answer,
                    computation_time=metrics_computation_time
                )
                
                # Log query execution
                duration = time.time() - start_time
                self.logger.log_query_execution(
                    question['question'], method, duration, True,
                    len(final_answer), len(contexts)
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                self.logger.log_query_execution(
                    question['question'], method, duration, False
                )
                
                # Return error result
                return self.data_processor.create_evaluation_result(
                    question, method, f"[ERROR: {str(e)}]", [], {}, str(e),
                    raw_answer="", processed_answer=""
                )
    
    async def _compute_metrics(self, question: Dict[str, Any], answer: str, 
                             contexts: List[str], metrics: List[str], method: str) -> Dict[str, float]:
        """Compute specified metrics for the evaluation in parallel using pooled adapters."""
        computed_metrics = {}
        answer_embedding = None
        ground_truth_embedding = None
        
        # Get question metadata for tracking
        question_id = question.get('id', question.get('question_id', 'unknown'))
        question_text = question.get('question', '')
        ground_truth = question.get('answer', '')
        
        # Create tasks for parallel execution, grouped by dependencies
        async def compute_single_metric(metric: str) -> tuple[str, float]:
            """Compute a single metric and return (metric_name, value)."""
            try:
                # LLM-based semantic metrics
                if metric == 'factual_accuracy_grade':
                    async with self.adapter_pool.get_llm_adapter_context(self.judge_llm_instance, "judge", self.smart_cache) as judge_adapter:
                        # Wrap with tracking
                        tracked_adapter = TrackedLLMAdapter(
                            llm_adapter=judge_adapter,
                            tracker=self.judge_tracker,
                            question_id=question_id,
                            question=question_text,
                            answer=answer,
                            ground_truth=ground_truth,
                            contexts=contexts,
                            default_metric='factual_accuracy_grade'
                        )
                        # Create a shared_results object with question_type for penalty system
                        class SharedResultsStub:
                            def __init__(self, question_type):
                                self.question_type = question_type
                                self.factual_accuracy_evaluation = None
                        
                        shared_results = SharedResultsStub(question.get('question_type', ''))
                        return metric, await self.semantic_metrics.compute_factual_accuracy_grade(
                            question['question'], answer, question.get('answer', ''), tracked_adapter, shared_results
                        )
                
                elif metric == 'semantic_similarity_percentage':
                    async with self.adapter_pool.get_embedding_adapter_context(self.embeddings_instance, "main", self.smart_cache) as embed_adapter:
                        return metric, await self.semantic_metrics.compute_semantic_similarity_percentage(
                            answer, question.get('answer', ''), embed_adapter
                        )
                
                # RAGAS metrics
                elif metric == 'ragas_faithfulness':
                    async with self.adapter_pool.get_llm_adapter_context(self.judge_llm_instance, "judge", self.smart_cache) as judge_adapter:
                        # Wrap with tracking
                        tracked_adapter = TrackedLLMAdapter(
                            llm_adapter=judge_adapter,
                            tracker=self.judge_tracker,
                            question_id=question_id,
                            question=question_text,
                            answer=answer,
                            ground_truth=ground_truth,
                            contexts=contexts,
                            default_metric='ragas_faithfulness'
                        )
                        score, details = await self.ragas_metrics.compute_faithfulness(
                            question['question'], answer, contexts, tracked_adapter
                        )
                        return metric, score
                
                elif metric == 'ragas_context_precision':
                    async with self.adapter_pool.get_llm_adapter_context(self.judge_llm_instance, "judge", self.smart_cache) as judge_adapter:
                        # Wrap with tracking
                        tracked_adapter = TrackedLLMAdapter(
                            llm_adapter=judge_adapter,
                            tracker=self.judge_tracker,
                            question_id=question_id,
                            question=question_text,
                            answer=answer,
                            ground_truth=ground_truth,
                            contexts=contexts,
                            default_metric='ragas_context_precision'
                        )
                        score, details = await self.ragas_metrics.compute_context_precision(
                            question['question'], contexts, question.get('answer', ''), 
                            tracked_adapter, reranker=self.reranker
                        )
                        return metric, score
                
                elif metric == 'ragas_context_recall':
                    async with self.adapter_pool.get_llm_adapter_context(self.judge_llm_instance, "judge", self.smart_cache) as judge_adapter:
                        gt_answer = question.get('answer', '')
                        if not gt_answer:
                            return metric, np.nan  # Cannot compute without ground truth (np.nan = missing data)
                        # Wrap with tracking
                        tracked_adapter = TrackedLLMAdapter(
                            llm_adapter=judge_adapter,
                            tracker=self.judge_tracker,
                            question_id=question_id,
                            question=question_text,
                            answer=answer,
                            ground_truth=gt_answer,
                            contexts=contexts,
                            default_metric='ragas_context_recall'
                        )
                        score, details = await self.ragas_metrics.compute_context_recall(
                            question['question'], contexts, gt_answer, tracked_adapter
                        )
                        return metric, score
                
                elif metric == 'ragas_answer_relevance':
                    async with self.adapter_pool.get_llm_adapter_context(self.judge_llm_instance, "judge", self.smart_cache) as judge_adapter:
                        async with self.adapter_pool.get_embedding_adapter_context(self.embeddings_instance, "main", self.smart_cache) as embed_adapter:
                            # Wrap with tracking
                            tracked_adapter = TrackedLLMAdapter(
                                llm_adapter=judge_adapter,
                                tracker=self.judge_tracker,
                                question_id=question_id,
                                question=question_text,
                                answer=answer,
                                ground_truth=ground_truth,
                                contexts=contexts,
                                default_metric='ragas_answer_relevance'
                            )
                            score, details = await self.ragas_metrics.compute_answer_relevance(
                                question['question'], answer, tracked_adapter, embed_adapter
                            )
                            return metric, score
                
                elif metric == 'ragas_score':
                    # This is a composite metric - will be computed after all individual RAGAS metrics
                    # Return a placeholder that will be replaced in the aggregation step
                    return metric, None  # None signals this needs post-computation
                
                # Text-based metrics (BERTScore)
                elif metric == 'bert_score_precision':
                    # BERTScore is CPU/GPU intensive but doesn't require API calls
                    # Run in executor to avoid blocking the event loop
                    import asyncio
                    loop = asyncio.get_event_loop()
                    score = await loop.run_in_executor(
                        None,  # Use default executor
                        self.text_metrics.compute_bert_score_precision,
                        answer,
                        question.get('answer', '')
                    )
                    return metric, score
                
                elif metric == 'bert_score_recall':
                    import asyncio
                    loop = asyncio.get_event_loop()
                    score = await loop.run_in_executor(
                        None,
                        self.text_metrics.compute_bert_score_recall,
                        answer,
                        question.get('answer', '')
                    )
                    return metric, score
                
                elif metric == 'bert_score_f1':
                    import asyncio
                    loop = asyncio.get_event_loop()
                    score = await loop.run_in_executor(
                        None,
                        self.text_metrics.compute_bert_score_f1,
                        answer,
                        question.get('answer', '')
                    )
                    return metric, score
                
                else:
                    self.logger.logger.warning(f"Unknown metric: {metric}, using fallback value")
                    return metric, 0.0
                
            except Exception as e:
                self.logger.log_metric_error(metric, question['question'], method, str(e))
                # Use fallback value from metric registry
                metric_config = self.metric_registry.get_metric_config(metric)
                fallback = metric_config.fallback_value if metric_config and metric_config.fallback_value is not None else 0.0
                return metric, fallback
        
        # Group metrics by independence for parallel execution
        independent_metrics = []
        classification_metrics_requested = []
        
        for metric in metrics:
            if metric in ['correct_answers_count', 'wrong_answers_count', 'dont_know_answers_count']:
                classification_metrics_requested.append(metric)
            else:
                independent_metrics.append(metric)
        
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
        
        # Post-process: Compute composite ragas_score from individual RAGAS metrics
        if 'ragas_score' in computed_metrics and computed_metrics.get('ragas_score') is None:
            computed_metrics['ragas_score'] = self._compute_composite_ragas_score(computed_metrics)
        
        # Handle answer classification metrics (they return multiple values)
        if classification_metrics_requested:
            try:
                async with self.adapter_pool.get_llm_adapter_context(self.judge_llm_instance, "judge", self.smart_cache) as judge_adapter:
                    # Wrap with tracking
                    tracked_adapter = TrackedLLMAdapter(
                        llm_adapter=judge_adapter,
                        tracker=self.judge_tracker,
                        question_id=question_id,
                        question=question_text,
                        answer=answer,
                        ground_truth=ground_truth,
                        contexts=contexts,
                        default_metric='answer_classification'
                    )
                    classification_results = await self.semantic_metrics.compute_answer_classification(
                        question['question'],
                        answer,
                        question.get('answer', ''),
                        tracked_adapter,
                        question_type=question.get('question_type')
                    )
                    # Add requested classification metrics to computed_metrics
                    for metric in classification_metrics_requested:
                        computed_metrics[metric] = classification_results.get(metric, 0.0)
            except Exception as e:
                self.logger.log_metric_error('answer_classification', question['question'], method, str(e))
                # Add fallback values for requested classification metrics
                for metric in classification_metrics_requested:
                    if metric not in computed_metrics:
                        computed_metrics[metric] = 0.0
        
        return computed_metrics
    
    def _compute_composite_ragas_score(self, metrics: Dict[str, float]) -> float:
        """
        Compute composite RAGAS score from individual RAGAS metrics.
        
        Uses weighted average with default weights:
        - Faithfulness: 0.3
        - Context Precision: 0.2  
        - Context Recall: 0.2
        - Answer Relevance: 0.3
        
        Args:
            metrics: Dictionary containing individual RAGAS metric values
            
        Returns:
            Weighted average RAGAS score (0-100), or 0.0 if no valid metrics
        """
        import numpy as np
        
        weights = {
            'ragas_faithfulness': 0.3,
            'ragas_context_precision': 0.2,
            'ragas_context_recall': 0.2,
            'ragas_answer_relevance': 0.3
        }
        
        valid_scores = {}
        for metric_name, weight in weights.items():
            value = metrics.get(metric_name)
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                valid_scores[metric_name] = value
        
        if not valid_scores:
            self.logger.logger.debug("No valid RAGAS metrics for composite score computation")
            return 0.0
        
        # Recalculate weights for valid scores only
        total_weight = sum(weights[k] for k in valid_scores)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(
            valid_scores[k] * weights[k] / total_weight
            for k in valid_scores
        )
        
        self.logger.logger.debug(
            f"Computed ragas_score={weighted_sum:.2f} from {len(valid_scores)} valid metrics: {list(valid_scores.keys())}"
        )
        
        return weighted_sum
    
    def _load_prompt_template(self, question_type: str) -> Optional[str]:
        """Load prompt template for question type."""
        try:
            template_file = Path(__file__).parent.parent / 'prompt_templates' / f"{question_type.lower().replace(' ', '_')}.txt"
            if template_file.exists():
                return template_file.read_text().strip()
        except Exception:
            pass
        return None
    
    def _save_intermediate_results(self):
        """Save intermediate results."""
        try:
            # Transfer query logs from logger to result processor
            for log_entry in self.logger.logs:
                if log_entry.get('type') == 'query_execution':
                    # Only add if not already present
                    if log_entry not in self.result_processor.query_log:
                        self.result_processor.add_query_log(log_entry)
            
            self.result_processor.save_results(self.config.output_dir)
            self.result_processor.save_metrics(self.config.output_dir)
            self.result_processor.save_query_log(self.config.output_dir)
            
            # Save judge evaluations (if enabled in config)
            if self.config.enable_judge_logging:
                self._save_judge_evaluations()
        except Exception as e:
            self.logger.logger.warning(f"Failed to save intermediate results: {str(e)}")
    
    def _save_final_results(self):
        """Save final results."""
        try:
            # Transfer query logs from logger to result processor (with deduplication)
            for log_entry in self.logger.logs:
                if log_entry.get('type') == 'query_execution':
                    # Only add if not already present (prevents duplicates from intermediate saves)
                    if log_entry not in self.result_processor.query_log:
                        self.result_processor.add_query_log(log_entry)
            
            self.result_processor.save_results(self.config.output_dir)
            self.result_processor.save_metrics(self.config.output_dir)
            self.result_processor.save_query_log(self.config.output_dir)
            
            # Save summary files for consistency with separated mode
            self.result_processor.save_benchmark_summary(self.config.output_dir, self.config)
            self.result_processor.save_metrics_summary(self.config.output_dir, self.config)
            
            # Save judge evaluations (if enabled in config)
            if self.config.enable_judge_logging:
                self._save_judge_evaluations()
            
            self.logger.save_logs()
        except Exception as e:
            self.logger.logger.error(f"Failed to save final results: {str(e)}")
    
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
    
    def _log_evaluation_summary(self):
        """Log evaluation summary including performance optimizations."""
        # Performance summary
        performance_summary = self.result_processor.get_performance_summary()
        self.logger.logger.info(f"Performance Summary: {performance_summary}")
        
        # Error summary
        error_summary = self.result_processor.get_error_summary()
        self.logger.logger.info(f"Error Summary: {error_summary}")
        
        # Cache statistics
        cache_stats = self.smart_cache.get_cache_stats()
        self.logger.logger.info(f"Cache Statistics: {cache_stats}")
        
        # Pool statistics
        pool_stats = self.adapter_pool.get_pool_stats()
        self.logger.logger.info(f"Adapter Pool Statistics: {pool_stats}")
        
        # Judge tracker statistics
        judge_summary = self.judge_tracker.get_summary()
        self.logger.logger.info(f"Judge LLM Interactions: {judge_summary}")
        
        # Log summary
        log_summary = self.logger.get_log_summary()
        self.logger.logger.info(f"Log Summary: {log_summary}")
        
        # Calculate performance gains from optimizations
        total_llm_requests = cache_stats['llm_cache']['hits'] + cache_stats['llm_cache']['misses']
        total_embedding_requests = cache_stats['embedding_cache']['hits'] + cache_stats['embedding_cache']['misses']
        
        if total_llm_requests > 0:
            llm_cache_benefit = (cache_stats['llm_cache']['hits'] / total_llm_requests) * 100
            self.logger.logger.info(f"LLM Cache provided {llm_cache_benefit:.1f}% reduction in API calls")
        
        if total_embedding_requests > 0:
            embed_cache_benefit = (cache_stats['embedding_cache']['hits'] / total_embedding_requests) * 100
            self.logger.logger.info(f"Embedding Cache provided {embed_cache_benefit:.1f}% reduction in API calls")
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all evaluation results."""
        return self.result_processor.get_results()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self.result_processor.get_performance_summary()
    
    def export_results(self, output_dir: str):
        """Export results to various formats."""
        self.result_processor.export_to_csv(output_dir) 
    
    def save_intermediate_results(self):
        """Public method to save intermediate results (called from main.py)."""
        self._save_intermediate_results() 
    
    async def cleanup(self):
        """Cleanup resources including adapters and sessions."""
        self.logger.logger.info("Cleaning up evaluator resources...")
        
        # Cleanup LLM adapters
        if self.llm_adapter and hasattr(self.llm_adapter, 'close'):
            try:
                await self.llm_adapter.close()
            except Exception as e:
                self.logger.logger.warning(f"Error cleaning up LLM adapter: {e}")
        
        if self.judge_llm_adapter and hasattr(self.judge_llm_adapter, 'close'):
            # Only cleanup if it's a different adapter than the main one
            if self.judge_llm_adapter != self.llm_adapter:
                try:
                    await self.judge_llm_adapter.close()
                except Exception as e:
                    self.logger.logger.warning(f"Error cleaning up judge LLM adapter: {e}")
        
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
        
        self.logger.logger.info("Evaluator cleanup complete") 