#!/usr/bin/env python3
"""
Separated Benchmark Runner - focuses only on data collection without metrics computation.
This allows for faster benchmarking and separate metrics computation.
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
from utils.logging_utils import EvaluationLogger
from utils.data_utils import DataLoader, DataProcessor
from adapters.model_manager import ModelManager
from adapters.llm_adapter import LLMAdapterFactory
from adapters.embedding_adapter import EmbeddingAdapterFactory
from core.query_runner import QueryRunnerFactory
from core.adapter_pool import AdapterPool
from core.smart_cache import SmartCache
from core.resource_monitor import ResourceMonitor
from core.context_builder import build_llm_context


class BenchmarkRunner:
    """Dedicated benchmark runner that focuses only on collecting raw evaluation data."""
    
    def __init__(self, config: EvaluationConfig, logger: EvaluationLogger):
        self.config = config
        self.logger = logger
        self.data_loader = DataLoader(logger.logger)
        self.data_processor = DataProcessor(logger.logger)
        self.model_manager = ModelManager(logger.logger)
        
        # Initialize adapter pool for better performance
        # REDUCED pool size to prevent connection exhaustion when running parallel benchmarks
        pool_size = max(config.max_concurrent_tasks // 4, 4)  # Smaller pool to reduce API pressure
        self.adapter_pool = AdapterPool(logger.logger, max_pool_size=pool_size)
        
        # Initialize smart cache with higher capacity for benchmarking
        cache_size = max(config.max_concurrent_tasks * 100, 2000)  # Larger cache
        self.smart_cache = SmartCache(logger.logger, max_cache_size=cache_size)
        
        # REDUCED concurrency to prevent server disconnection errors when running parallel benchmarks
        # Original: config.max_concurrent_tasks * 3, now more conservative
        self.semaphore = asyncio.Semaphore(max(config.max_concurrent_tasks, 6))
        
        # Add rate limiting delay between batches (seconds)
        self.batch_delay = 0.5  # Small delay to prevent API overwhelm
        
        # Model components
        self.llm_adapter = None
        self.embeddings_adapter = None
        self.llm_instance = None
        self.embeddings_instance = None
        self.graphrag_config = None
        self.index_files = {}
        self.novel_contexts = {}
        
        # Results storage
        self.benchmark_results: List[Dict[str, Any]] = []
        self.query_log: List[Dict[str, Any]] = []  # Track query execution logs
        self.resource_monitor = ResourceMonitor(logger.logger)
        
        # Checkpointing support with enhanced granular tracking
        self.checkpoint_file = Path(config.output_dir) / "benchmark_checkpoint.json"
        self.checkpoint_interval = 10  # Save checkpoint every N tasks
        self.processed_tasks: set = set()  # Track completed (question_id, method) pairs
        # Enhanced checkpoint metadata for better tracking and debugging
        self.checkpoint_metadata: Dict[str, Dict[str, Any]] = {}  # Maps task_id to metadata
    
    async def initialize(self):
        """Initialize the benchmark runner with models and data."""
        self.logger.logger.info("Initializing benchmark runner...")
        
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
        
        # Load novel contexts asynchronously
        if self.config.input_json_path:
            self.novel_contexts = await self.data_loader.load_novel_contexts_async(self.config.input_json_path)
        
        self.logger.logger.info("Benchmark runner initialization complete")
    
    async def run_benchmark(self, questions_path: str, resume_from_checkpoint: bool = False) -> str:
        """Run the benchmarking process and return path to raw results file."""
        self.logger.logger.info(f"Starting benchmark data collection with {len(self.config.methods)} methods")
        
        # Load checkpoint if resuming
        if resume_from_checkpoint and self.checkpoint_file.exists():
            self._load_checkpoint()
            self.logger.logger.info(f"Resuming from checkpoint with {len(self.processed_tasks)} completed tasks")
        
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
            'max_samples_per_type': self.config.max_samples_per_type,
            'benchmark_mode': True
        })
        
        # Create benchmark tasks
        all_tasks = []
        task_info = []
        # CRITICAL-002 FIX: Store deep copies of original questions to preserve data in error handling
        from copy import deepcopy
        all_questions = []
        
        for q_type, questions_list in grouped_questions.items():
            for question in questions_list:
                for method in self.config.methods:
                    # Create unique task identifier
                    task_id = f"{question.get('id', question.get('question', '')[:50])}_{method}"
                    
                    # Skip if already processed (checkpoint resume)
                    if task_id in self.processed_tasks:
                        self.logger.logger.debug(f"Skipping already processed task: {task_id}")
                        continue
                    
                    task = self._benchmark_single_question(question, method, q_type)
                    all_tasks.append(task)
                    task_info.append((q_type, method, task_id))
                    # CRITICAL-002 FIX: Store deep copy to ensure original data is preserved
                    all_questions.append(deepcopy(question))

        # Execute tasks with optimized batching for pure benchmarking
        total_tasks = len(all_tasks)
        completed = 0
        
        # REDUCED batch size to prevent server overwhelm during parallel execution
        batch_size = min(self.config.batch_size * 4, self.config.max_concurrent_tasks)
        
        self.logger.logger.info(f"Processing {total_tasks} benchmark tasks in batches of {batch_size}")
        
        try:
            for batch_start in range(0, total_tasks, batch_size):
                batch_end = min(batch_start + batch_size, total_tasks)
                batch_tasks = all_tasks[batch_start:batch_end]
                
                try:
                    # Execute batch concurrently
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Add small delay between batches to prevent API overwhelm
                    if hasattr(self, 'batch_delay') and self.batch_delay > 0:
                        await asyncio.sleep(self.batch_delay)
                    
                    # Process results
                    for i, result in enumerate(batch_results):
                        task_idx = batch_start + i
                        if isinstance(result, Exception):
                            # Enhanced error logging with full context
                            if task_idx < len(task_info):
                                q_type, method, task_id = task_info[task_idx]
                            else:
                                q_type, method, task_id = 'unknown', 'unknown', f'task_{task_idx}'
                            
                            self.logger.logger.error(f"Benchmark task failed for task {task_idx}:")
                            self.logger.logger.error(f"  Question Type: {q_type}")
                            self.logger.logger.error(f"  Method: {method}")
                            self.logger.logger.error(f"  Error: {str(result)}")
                            
                            # Log original question data for context
                            original_question = all_questions[task_idx] if task_idx < len(all_questions) else None
                            if original_question:
                                self.logger.logger.error(f"  Question: {original_question.get('question', 'N/A')[:200]}...")
                                self.logger.logger.error(f"  Question ID: {original_question.get('id', 'N/A')}")
                                self.logger.logger.error(f"  Source: {original_question.get('source', 'N/A')}")
                            
                            error_result = self._create_error_benchmark_result(task_idx, q_type, method, str(result), original_question)
                            self.benchmark_results.append(error_result)
                        elif isinstance(result, dict):
                            self.benchmark_results.append(result)
                            # Track completed task for checkpoint with enhanced metadata
                            if task_idx < len(task_info):
                                q_type, method, task_id = task_info[task_idx]
                                self.processed_tasks.add(task_id)
                                # Store enhanced checkpoint metadata
                                self.checkpoint_metadata[task_id] = {
                                    'question_type': q_type,
                                    'method': method,
                                    'question_id': result.get('id', 'unknown'),
                                    'source': result.get('source', 'unknown'),
                                    'timestamp': result.get('timestamp', datetime.now(timezone.utc).isoformat()),
                                    'contexts_count': len(result.get('contexts', [])),
                                    'answer_length': len(result.get('final_answer', ''))
                                }
                                # Log progress for successful results
                                self.logger.log_evaluation_progress(completed + i + 1, total_tasks, q_type, method)
                    
                    # Update completed count
                    completed += len(batch_tasks)
                    
                    # Log progress
                    progress_pct = (completed / total_tasks) * 100
                    self.logger.logger.info(f"Benchmark Progress: {completed}/{total_tasks} ({progress_pct:.1f}%)")
                    
                    # Save intermediate results more frequently
                    if completed % batch_size == 0:
                        self._save_intermediate_benchmark_results()
                    
                    # Save checkpoint at intervals
                    if completed % self.checkpoint_interval == 0:
                        self._save_checkpoint()
                    
                except Exception as e:
                    self.logger.logger.error(f"Benchmark batch execution failed: {str(e)}")
                    # Create error results for failed batch
                    for i in range(len(batch_tasks)):
                        task_idx = batch_start + i
                        q_type, method = task_info[task_idx] if task_idx < len(task_info) else ('unknown', 'unknown')
                        # Preserve original question data in error result
                        original_question = all_questions[task_idx] if task_idx < len(all_questions) else None
                        error_result = self._create_error_benchmark_result(task_idx, q_type, method, str(e), original_question)
                        self.benchmark_results.append(error_result)
                    completed += len(batch_tasks)
                    
                    # Save intermediate results even after batch failure
                    self._save_intermediate_benchmark_results()
                    
        except KeyboardInterrupt:
            self.logger.logger.info(f"Benchmark interrupted by user after {completed}/{total_tasks} tasks")
            # Save all intermediate results and checkpoint before re-raising
            self._save_intermediate_benchmark_results()
            self._save_checkpoint()
            self.logger.logger.info(f"Progress saved to checkpoint: {self.checkpoint_file}")
            self.logger.logger.info(f"To resume, run with --resume_checkpoint flag")
            raise
        
        # Final save
        raw_results_path = self._save_final_benchmark_results()
        
        # Clear checkpoint after successful completion
        self.clear_checkpoint()
        
        # Log benchmark summary
        self._log_benchmark_summary()
        
        self.logger.logger.info("Benchmark data collection complete")
        return raw_results_path
    
    async def _benchmark_single_question(self, question: Dict[str, Any], method: str, question_type: str) -> Dict[str, Any]:
        """Benchmark a single question with a specific method - no metrics computation."""
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
                # Use shared context builder to ensure consistency with evaluator mode
                # This includes ALL documents from novel.json/medical.json, not just the source
                context = None
                if method == 'llm_with_context':
                    source = question.get('source', '')
                    context = build_llm_context(self.novel_contexts, source, self.logger.logger)
                
                # Run query
                raw_answer, processed_answer, contexts = await runner.run_query(
                    question['question'], context, prompt_template
                )
                
                # Determine final answer based on whether we have prompt template
                final_answer = processed_answer if prompt_template else raw_answer
                
                # Create benchmark result (no metrics computed here)
                result = self.data_processor.create_benchmark_result(
                    question, method, final_answer, contexts,
                    raw_answer=raw_answer, processed_answer=processed_answer
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
                # Enhanced error logging with full context
                self.logger.logger.error(f"Query execution failed:")
                self.logger.logger.error(f"  Question: {question.get('question', 'N/A')[:200]}...")
                self.logger.logger.error(f"  Method: {method}")
                self.logger.logger.error(f"  Question Type: {question_type}")
                self.logger.logger.error(f"  Error: {str(e)}")
                self.logger.logger.error(f"  Duration: {duration:.2f}s")
                
                self.logger.log_query_execution(
                    question['question'], method, duration, False
                )
                
                # Return error result
                return self.data_processor.create_benchmark_result(
                    question, method, f"[ERROR: {str(e)}]", [], 
                    error_message=str(e), raw_answer="", processed_answer=""
                )
    
    def _create_error_benchmark_result(self, task_idx: int, q_type: str, method: str, error_msg: str, 
                                       original_question: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create error benchmark result, preserving original question data when available."""
        if original_question:
            error_question = {
                'id': original_question.get('id', f'benchmark_error_{task_idx}'),
                'question': original_question.get('question', 'Benchmark task failed'),
                'question_type': original_question.get('question_type', q_type),
                'answer': original_question.get('answer'),
                'source': original_question.get('source', 'unknown'),
                'evidence': original_question.get('evidence', []),
                'evidence_tripples': original_question.get('evidence_tripples', [])
            }
        else:
            error_question = {
                'id': f'benchmark_error_{task_idx}', 
                'question': 'Benchmark task failed', 
                'question_type': q_type,
                'answer': None,
                'source': 'unknown',
                'evidence': [],
                'evidence_tripples': []
            }
        return self.data_processor.create_benchmark_result(
            error_question, method, f"[BENCHMARK ERROR: {error_msg}]", [],
            error_message=error_msg, raw_answer="", processed_answer=""
        )
    
    def _load_prompt_template(self, question_type: str) -> Optional[str]:
        """Load prompt template for question type."""
        try:
            template_file = Path(__file__).parent.parent / 'prompt_templates' / f"{question_type.lower().replace(' ', '_')}.txt"
            if template_file.exists():
                return template_file.read_text().strip()
        except Exception:
            pass
        return None
    
    def _save_intermediate_benchmark_results(self):
        """Save intermediate benchmark results."""
        try:
            output_path = Path(self.config.output_dir) / "benchmark_raw_results.json"
            with open(output_path, 'w') as f:
                json.dump(self.benchmark_results, f, indent=2, default=str)
            self.logger.logger.debug(f"Intermediate benchmark results saved to {output_path}")
        except Exception as e:
            self.logger.logger.warning(f"Failed to save intermediate benchmark results: {str(e)}")
    
    def _save_final_benchmark_results(self) -> str:
        """Save final benchmark results and return path."""
        try:
            output_path = Path(self.config.output_dir) / "benchmark_raw_results.json"
            with open(output_path, 'w') as f:
                json.dump(self.benchmark_results, f, indent=2, default=str)
            
            # Also save a summary for quick overview
            summary_path = Path(self.config.output_dir) / "benchmark_summary.json"
            summary = {
                'total_results': len(self.benchmark_results),
                'methods': list(set(r.get('method', 'unknown') for r in self.benchmark_results)),
                'question_types': list(set(r.get('question_type', 'unknown') for r in self.benchmark_results)),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'config': {
                    'max_concurrent_tasks': self.config.max_concurrent_tasks,
                    'batch_size': self.config.batch_size,
                    'cache_enabled': True
                }
            }
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.logger.info(f"Final benchmark results saved to {output_path}")
            self.logger.logger.info(f"Benchmark summary saved to {summary_path}")
            
            # Save query log
            self._save_query_log()
            
            return str(output_path)
            
        except Exception as e:
            self.logger.logger.error(f"Failed to save final benchmark results: {str(e)}")
            raise
    
    def _save_query_log(self):
        """Save query execution log."""
        try:
            query_log_path = Path(self.config.output_dir) / "query.json"
            
            # Transfer query logs from logger
            for log_entry in self.logger.logs:
                if log_entry.get('type') == 'query_execution':
                    if log_entry not in self.query_log:
                        self.query_log.append(log_entry)
            
            with open(query_log_path, 'w') as f:
                json.dump(self.query_log, f, indent=2, default=str)
            
            self.logger.logger.info(f"Query log saved to {query_log_path}")
            
        except Exception as e:
            self.logger.logger.warning(f"Failed to save query log: {str(e)}")
    
    def _log_benchmark_summary(self):
        """Log benchmark summary including performance statistics."""
        # Cache statistics
        cache_stats = self.smart_cache.get_cache_stats()
        self.logger.logger.info(f"Benchmark Cache Statistics: {cache_stats}")
        
        # Pool statistics
        pool_stats = self.adapter_pool.get_pool_stats()
        self.logger.logger.info(f"Benchmark Adapter Pool Statistics: {pool_stats}")
        
        # Results summary
        total_results = len(self.benchmark_results)
        error_results = len([r for r in self.benchmark_results if r.get('error_message')])
        success_rate = ((total_results - error_results) / total_results * 100) if total_results > 0 else 0
        
        self.logger.logger.info(f"Benchmark Results Summary:")
        self.logger.logger.info(f"  Total results: {total_results}")
        self.logger.logger.info(f"  Error results: {error_results}")
        self.logger.logger.info(f"  Success rate: {success_rate:.1f}%")
        
        # Performance gains from optimizations
        total_llm_requests = cache_stats['llm_cache']['hits'] + cache_stats['llm_cache']['misses']
        if total_llm_requests > 0:
            llm_cache_benefit = (cache_stats['llm_cache']['hits'] / total_llm_requests) * 100
            self.logger.logger.info(f"Benchmark LLM Cache provided {llm_cache_benefit:.1f}% reduction in API calls")
    
    def get_benchmark_results(self) -> List[Dict[str, Any]]:
        """Get all benchmark results."""
        return self.benchmark_results.copy()
    
    def _save_checkpoint(self):
        """Save checkpoint with enhanced granular progress tracking."""
        try:
            # Build metadata list from checkpoint_metadata dictionary
            metadata_list = []
            for task_id, meta in self.checkpoint_metadata.items():
                metadata_list.append({
                    'task_id': task_id,
                    'question_type': meta.get('question_type', 'unknown'),
                    'method': meta.get('method', 'unknown'),
                    'question_id': meta.get('question_id', 'unknown'),
                    'source': meta.get('source', 'unknown'),
                    'timestamp': meta.get('timestamp', datetime.now(timezone.utc).isoformat()),
                    'contexts_count': meta.get('contexts_count', 0),
                    'answer_length': meta.get('answer_length', 0)
                })
            
            # Group by different dimensions for analysis
            by_question_type = defaultdict(list)
            by_method = defaultdict(list)
            by_source = defaultdict(list)
            
            for meta in metadata_list:
                by_question_type[meta['question_type']].append(meta['task_id'])
                by_method[meta['method']].append(meta['task_id'])
                by_source[meta['source']].append(meta['task_id'])
            
            checkpoint_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'processed_tasks': list(self.processed_tasks),
                'total_completed': len(self.benchmark_results),
                # Enhanced granular tracking
                'metadata': metadata_list,
                'summary_by_question_type': {k: len(v) for k, v in by_question_type.items()},
                'summary_by_method': {k: len(v) for k, v in by_method.items()},
                'summary_by_source': {k: len(v) for k, v in by_source.items()},
                'error_count': len([r for r in self.benchmark_results if r.get('error_message')])
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            self.logger.logger.debug(
                f"Checkpoint saved: {len(self.processed_tasks)} tasks processed "
                f"({checkpoint_data['error_count']} errors)"
            )
        except Exception as e:
            self.logger.logger.warning(f"Failed to save checkpoint: {str(e)}")
    
    def _load_checkpoint(self):
        """Load checkpoint with enhanced granular tracking to resume computation."""
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.processed_tasks = set(checkpoint_data.get('processed_tasks', []))
            
            # Load enhanced metadata if available
            if 'metadata' in checkpoint_data:
                for meta in checkpoint_data['metadata']:
                    task_id = meta.get('task_id', '')
                    if task_id:
                        self.checkpoint_metadata[task_id] = {
                            'question_type': meta.get('question_type', 'unknown'),
                            'method': meta.get('method', 'unknown'),
                            'question_id': meta.get('question_id', 'unknown'),
                            'source': meta.get('source', 'unknown'),
                            'timestamp': meta.get('timestamp', 'unknown'),
                            'contexts_count': meta.get('contexts_count', 0),
                            'answer_length': meta.get('answer_length', 0)
                        }
                self.logger.logger.info(f"Loaded metadata for {len(self.checkpoint_metadata)} processed tasks")
            
            # Try to load intermediate results if they exist
            intermediate_path = Path(self.config.output_dir) / "benchmark_raw_results.json"
            if intermediate_path.exists():
                with open(intermediate_path, 'r') as f:
                    self.benchmark_results = json.load(f)
                
                # Rebuild checkpoint_metadata from benchmark_results if not loaded from checkpoint
                if not self.checkpoint_metadata:
                    for result in self.benchmark_results:
                        question_id = result.get('id', 'unknown')
                        method = result.get('method', 'unknown')
                        task_id = f"{question_id}_{method}"
                        if task_id in self.processed_tasks:
                            self.checkpoint_metadata[task_id] = {
                                'question_type': result.get('question_type', 'unknown'),
                                'method': method,
                                'question_id': question_id,
                                'source': result.get('source', 'unknown'),
                                'timestamp': result.get('timestamp', 'unknown'),
                                'contexts_count': len(result.get('contexts', [])),
                                'answer_length': len(result.get('final_answer', ''))
                            }
                
                self.logger.logger.info(f"Loaded {len(self.benchmark_results)} benchmark results from checkpoint")
            
            checkpoint_time = checkpoint_data.get('timestamp', 'unknown')
            self.logger.logger.info(f"Checkpoint loaded from {checkpoint_time}")
            
            # Log enhanced summary information
            if 'summary_by_question_type' in checkpoint_data:
                self.logger.logger.info(f"  By question type: {checkpoint_data['summary_by_question_type']}")
            if 'summary_by_method' in checkpoint_data:
                self.logger.logger.info(f"  By method: {checkpoint_data['summary_by_method']}")
            if 'error_count' in checkpoint_data:
                self.logger.logger.info(f"  Total errors: {checkpoint_data['error_count']}")
            
            self.logger.logger.info(f"Resuming with {len(self.processed_tasks)} completed tasks")
        except Exception as e:
            self.logger.logger.error(f"Failed to load checkpoint: {str(e)}")
            self.logger.logger.info("Starting fresh benchmark run")
    
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
        self.logger.logger.info("Cleaning up benchmark runner resources...")
        
        # Cleanup LLM adapter
        if self.llm_adapter and hasattr(self.llm_adapter, 'close'):
            try:
                await self.llm_adapter.close()
            except Exception as e:
                self.logger.logger.warning(f"Error cleaning up LLM adapter: {e}")
        
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
        
        self.logger.logger.info("Benchmark runner cleanup complete")
