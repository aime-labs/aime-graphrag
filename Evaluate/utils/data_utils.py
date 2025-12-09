import json
import pandas as pd
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from .error_handling import validate_input_data, sanitize_text


class DataLoader:
    """Utility class for loading and validating evaluation data."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._context_cache = {}  # Cache for novel contexts
    
    def load_questions(self, questions_path: str) -> List[Dict[str, Any]]:
        """Load questions from JSON file with validation."""
        try:
            with open(questions_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("Questions file must contain a list of question objects")
            
            # Validate each question
            validated_questions = []
            for i, question in enumerate(data):
                if self._validate_question(question, i):
                    validated_questions.append(question)
            
            self.logger.info(f"Loaded {len(validated_questions)} valid questions from {questions_path}")
            return validated_questions
            
        except Exception as e:
            self.logger.error(f"Failed to load questions from {questions_path}: {str(e)}")
            raise
    
    async def load_questions_async(self, questions_path: str, 
                                   question_types_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Load questions asynchronously with optional type filtering.
        
        Args:
            questions_path: Path to questions JSON file
            question_types_filter: Optional list of question types to include (filters out others)
        """
        # Run the synchronous load in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        questions = await loop.run_in_executor(None, self.load_questions, questions_path)
        
        # Apply question type filter if specified
        if question_types_filter:
            filtered_questions = [
                q for q in questions 
                if q.get('question_type', 'Uncategorized') in question_types_filter
            ]
            self.logger.info(
                f"Filtered questions by type {question_types_filter}: "
                f"{len(questions)} -> {len(filtered_questions)}"
            )
            return filtered_questions
        
        return questions
    
    async def load_novel_contexts_async(self, input_json_path: str) -> Dict[str, str]:
        """Load novel contexts asynchronously with caching."""
        if input_json_path in self._context_cache:
            self.logger.debug(f"Using cached contexts for {input_json_path}")
            return self._context_cache[input_json_path]
        
        # Load contexts in executor
        loop = asyncio.get_event_loop()
        contexts = await loop.run_in_executor(None, self.load_novel_contexts, input_json_path)
        
        # Cache the contexts
        self._context_cache[input_json_path] = contexts
        return contexts
    
    async def preload_contexts_for_questions(self, questions: List[Dict[str, Any]], 
                                           novel_contexts: Dict[str, str]) -> None:
        """Preload and cache contexts for questions that need them."""
        # Group questions by source to batch context loading
        sources_needed = set()
        for question in questions:
            source = question.get('source')
            if source and str(source) in novel_contexts:
                sources_needed.add(str(source))
        
        if sources_needed:
            self.logger.info(f"Preloaded contexts for {len(sources_needed)} unique sources")
    
    def _validate_question(self, question: Dict[str, Any], index: int) -> bool:
        """Validate a single question object."""
        required_fields = ['id', 'question', 'answer']
        
        if not validate_input_data(question, required_fields, self.logger):
            self.logger.warning(f"Skipping question {index}: missing required fields")
            return False
        
        # Sanitize text fields - preserve full text for ground truth (answer) to ensure
        # accurate metric computation. Truncation would corrupt RAGAS and other metrics.
        question['question'] = sanitize_text(question.get('question', ''), truncate=False)
        question['answer'] = sanitize_text(question.get('answer', ''), truncate=False)
        
        # Set default values for optional fields
        question.setdefault('question_type', 'Uncategorized')
        question.setdefault('source', 'unknown')
        question.setdefault('evidence', [])
        question.setdefault('evidence_tripples', [])
        
        # Validate question type
        valid_types = [
            'Fact Retrieval', 'Complex Reasoning', 'Contextual Summarize', 
            'Creative Generation', 'Retrieval', 'Uncategorized'
        ]
        if question['question_type'] not in valid_types:
            self.logger.warning(
                f"Question {index}: invalid question_type '{question['question_type']}', "
                f"using 'Uncategorized'"
            )
            question['question_type'] = 'Uncategorized'
        
        return True
    
    def _clean_context(self, text: str) -> str:
        """Clean context text without truncation.
        
        Performs basic cleanup (remove null bytes, normalize line endings)
        but preserves full text length for LLM context evaluation.
        The AIME API supports up to 120K tokens.
        """
        if not isinstance(text, str):
            return ""
        
        # Remove null bytes and other problematic characters
        text = text.replace('\x00', '')
        text = text.replace('\r\n', '\n')  # Normalize line endings
        text = text.replace('\r', '\n')
        
        # Don't truncate - full context needed for llm_with_context evaluation
        return text
    
    def load_novel_contexts(self, input_json_path: Optional[str]) -> Dict[str, str]:
        """Load novel contexts from JSON file.
        
        Note: Novel contexts are NOT truncated here - they need full length for
        llm_with_context evaluation. The LLM API supports up to 120K tokens.
        Only basic cleanup (null bytes, normalize whitespace) is performed.
        """
        contexts = {}
        
        if not input_json_path:
            return contexts
        
        try:
            with open(input_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for entry in data:
                    corpus_name = entry.get('corpus_name')
                    context = entry.get('context', '')
                    if corpus_name:
                        # Clean but DON'T truncate - full context needed for evaluation
                        contexts[corpus_name] = self._clean_context(context)
            elif isinstance(data, dict):
                corpus_name = data.get('corpus_name')
                context = data.get('context', '')
                if corpus_name:
                    # Clean but DON'T truncate - full context needed for evaluation
                    contexts[corpus_name] = self._clean_context(context)
            
            self.logger.info(f"Loaded {len(contexts)} novel contexts from {input_json_path}")
            return contexts
            
        except Exception as e:
            self.logger.warning(f"Failed to load novel contexts from {input_json_path}: {str(e)}")
            return contexts
    
    def group_questions_by_type(self, questions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group questions by question type."""
        grouped = {}
        
        for question in questions:
            q_type = question.get('question_type', 'Uncategorized')
            if q_type not in grouped:
                grouped[q_type] = []
            grouped[q_type].append(question)
        
        # Log grouping statistics
        for q_type, items in grouped.items():
            self.logger.info(f"Question type '{q_type}': {len(items)} questions")
        
        return grouped


class DataProcessor:
    """Utility class for processing and transforming evaluation data."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def truncate_context(self, context: Any, max_length: int = 300) -> str:
        """Truncate context string for display."""
        if context is None:
            return ""
        
        text = str(context)
        if len(text) <= max_length:
            return text
        
        return text[:max_length] + "..."
    
    def prepare_context_list(self, context: Any) -> List[str]:
        """Convert context to list of strings."""
        if context is None:
            return []
        elif isinstance(context, list):
            return [str(x) for x in context]
        else:
            return [str(context)]
    
    def extract_evidence_triples(self, question: Dict[str, Any]) -> List[str]:
        """Extract evidence triples from question."""
        return question.get('evidence_tripples', [])
    
    def extract_evidence(self, question: Dict[str, Any]) -> List[str]:
        """Extract evidence from question."""
        return question.get('evidence', [])
    
    def create_evaluation_result(self, question: Dict[str, Any], method: str, 
                               answer: str, contexts: List[str], metrics: Dict[str, Any],
                               error_message: str = "", full_query: str = "", 
                               raw_answer: str = "", processed_answer: str = "",
                               answer_embedding: List[float] = None, 
                               ground_truth_embedding: List[float] = None,
                               computation_time: float = None) -> Dict[str, Any]:
        """Create a standardized evaluation result with all fields needed for metric computation."""
        # Use processed_answer if provided, otherwise fall back to answer
        final_answer = processed_answer if processed_answer else answer
        
        result = {
            # Core identification fields
            'question_id': question.get('id'),
            'question': question.get('question'),
            'gold_answer': question.get('answer'),
            'question_type': question.get('question_type', 'Uncategorized'),
            'source': question.get('source', 'unknown'),
            'method': method,
            
            # Answer fields (different formats for analysis)
            'raw_answer': raw_answer if raw_answer else answer,  # Raw response from GraphRAG
            'answer': final_answer,  # Processed answer using prompt templates
            'processed_answer': processed_answer if processed_answer else answer,  # Explicit processed answer
            
            # Context and retrieval information (standardized field names)
            'contexts': contexts,  # Full contexts for RAGAS and retrieval metrics
            'contexts_preview': [self.truncate_context(c) for c in contexts],  # Truncated for display
            'context_count': len(contexts),
            
            # Computed metrics
            'metrics': metrics,
            
            # Evidence and fact checking fields
            'evidence': self.extract_evidence(question),
            'evidence_tripples': self.extract_evidence_triples(question),
            
            # Embeddings for similarity analysis
            'answer_embedding': answer_embedding if answer_embedding is not None else [],
            'ground_truth_embedding': ground_truth_embedding if ground_truth_embedding is not None else [],
            
            # Meta fields
            'error_message': error_message,
            'full_query': full_query,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            
            # Additional analysis fields
            'answer_length': len(final_answer) if final_answer else 0,
            'question_length': len(question.get('question', '')),
            'ground_truth_length': len(question.get('answer', '')),
            
            # Method-specific metadata
            'architecture': self._determine_architecture(method),
            'is_retrieval_method': method in ['local_search', 'global_search', 'basic_search'],
            'is_llm_only_method': method in ['llm_with_context'],
        }
        
        # Add computation_time if provided (for metrics computation tracking)
        if computation_time is not None:
            result['computation_time'] = computation_time
            
        return result
    
    def _determine_architecture(self, method: str) -> str:
        """Determine the architecture type based on method."""
        if method in ['local_search', 'global_search']:
            return 'graphrag'
        elif method == 'basic_search':
            return 'rag'
        elif method in ['llm_with_context']:
            return 'direct_llm'
        else:
            return 'unknown'

    def create_benchmark_result(self, question: Dict[str, Any], method: str, 
                              final_answer: str, contexts: List[str], 
                              raw_answer: str = "", processed_answer: str = "",
                              error_message: str = "") -> Dict[str, Any]:
        """Create a standardized benchmark result without metrics.
        
        FIX: CRITICAL-003 - Store full contexts for proper metrics computation.
        The 'contexts_preview' field contains truncated contexts for display only,
        while 'contexts' contains full contexts for RAGAS and other metrics.
        """
        return {
            'id': question.get('id'),
            'question': question.get('question'),
            'question_type': question.get('question_type', 'Uncategorized'),
            'source': question.get('source', 'unknown'),
            'method': method,
            'final_answer': final_answer,
            # FIX: CRITICAL-003 - Store FULL contexts for metrics computation
            'contexts': contexts,  # Full contexts for RAGAS and retrieval metrics
            'contexts_preview': [self.truncate_context(c) for c in contexts],  # Truncated for display only
            'raw_answer': raw_answer,
            'processed_answer': processed_answer,
            'error_message': error_message,
            'ground_truth_answer': question.get('answer', ''),
            'evidence': self.extract_evidence(question),
            'evidence_tripples': self.extract_evidence_triples(question),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def create_metric_result(self, question: Dict[str, Any], method: str, 
                           metrics: Dict[str, Any], error_message: str = "") -> Dict[str, Any]:
        """Create a standardized metric result with validation."""
        # Validate and sanitize metrics before storing
        validated_metrics = self._validate_metrics(metrics)
        
        return {
            'question_id': question.get('id'),
            'question': question.get('question'),
            'question_type': question.get('question_type', 'Uncategorized'),
            'source': question.get('source', 'unknown'),
            'method': method,
            'metrics': validated_metrics,
            'error_message': error_message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _validate_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize metric values.
        
        Validation rules:
        - Percentage metrics (ending in '_percentage', '_score', 'bert_score_f1'): 0-100 range
        - Count metrics (ending in '_count'): >= 0
        - Grade metrics (factual_accuracy_grade): A/B/C/D/E/N/A/ERROR
        - RAGAS metrics: 0-100 range
        - NaN values are preserved (indicate computation failure)
        """
        import math
        
        validated = {}
        
        for metric_name, value in metrics.items():
            # Skip None values
            if value is None:
                validated[metric_name] = value
                continue
            
            # Handle NaN - preserve it as indicator of failed computation
            if isinstance(value, float) and math.isnan(value):
                validated[metric_name] = value
                continue
            
            # Validate grade metrics (string values like A, B, C, D, E)
            if metric_name == 'factual_accuracy_grade':
                valid_grades = {'A', 'B', 'C', 'D', 'E', 'N/A', 'ERROR'}
                if isinstance(value, str) and value.upper() in valid_grades:
                    validated[metric_name] = value.upper()
                else:
                    # Invalid grade - mark as error
                    validated[metric_name] = 'ERROR'
                continue
            
            # Validate numeric metrics
            if isinstance(value, (int, float)):
                # Percentage and score metrics: clamp to 0-100
                if any(suffix in metric_name for suffix in ['_percentage', '_score', 'bert_score', 'ragas_']):
                    # Clamp to valid range and log warning if out of bounds
                    if value < 0 or value > 100:
                        # Already clamped at source, but double-check
                        value = max(0.0, min(100.0, value))
                    validated[metric_name] = value
                
                # Count metrics: ensure non-negative
                elif '_count' in metric_name:
                    validated[metric_name] = max(0.0, value)
                
                # Other numeric metrics: keep as-is
                else:
                    validated[metric_name] = value
            else:
                # Non-numeric, non-grade value - keep as-is but log
                validated[metric_name] = value
        
        return validated
    
    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics across multiple results."""
        if not results:
            return {}
        
        # Collect all metric names
        all_metrics = set()
        for result in results:
            if 'metrics' in result and isinstance(result['metrics'], dict):
                all_metrics.update(result['metrics'].keys())
        
        aggregated = {}
        for metric in all_metrics:
            values = []
            for result in results:
                if 'metrics' in result and isinstance(result['metrics'], dict):
                    value = result['metrics'].get(metric)
                    if value is not None and isinstance(value, (int, float)):
                        values.append(value)
            
            if values:
                aggregated[metric] = sum(values) / len(values)
        
        return aggregated
    
    def filter_results_by_type(self, results: List[Dict[str, Any]], 
                              question_type: str) -> List[Dict[str, Any]]:
        """Filter results by question type."""
        return [
            result for result in results 
            if result.get('question_type') == question_type
        ]
    
    def filter_results_by_method(self, results: List[Dict[str, Any]], 
                                method: str) -> List[Dict[str, Any]]:
        """Filter results by method."""
        return [
            result for result in results 
            if result.get('method') == method
        ]
    
    def get_performance_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance summary from results."""
        if not results:
            return {}
        
        # Group by question type and method
        summary = {}
        
        for result in results:
            q_type = result.get('question_type', 'Uncategorized')
            method = result.get('method', 'unknown')
            
            if q_type not in summary:
                summary[q_type] = {}
            
            if method not in summary[q_type]:
                summary[q_type][method] = {
                    'count': 0,
                    'errors': 0,
                    'metrics': {}
                }
            
            summary[q_type][method]['count'] += 1
            
            if result.get('error_message'):
                summary[q_type][method]['errors'] += 1
            
            # Aggregate metrics
            if 'metrics' in result and isinstance(result['metrics'], dict):
                for metric, value in result['metrics'].items():
                    if value is not None and isinstance(value, (int, float)):
                        if metric not in summary[q_type][method]['metrics']:
                            summary[q_type][method]['metrics'][metric] = []
                        summary[q_type][method]['metrics'][metric].append(value)
        
        # Calculate averages
        for q_type in summary:
            for method in summary[q_type]:
                metrics = summary[q_type][method]['metrics']
                for metric in metrics:
                    values = metrics[metric]
                    if values:
                        metrics[metric] = sum(values) / len(values)
        
        return summary 