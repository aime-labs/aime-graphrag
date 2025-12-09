import asyncio
import numpy as np
import warnings
from typing import Any, Dict, List, Optional
import logging
from .optimized_llm_processor import OptimizedLLMProcessor, SharedLLMResults
from .hallucination_metrics import HallucinationMetrics
from .semantic_metrics import SemanticMetrics
from .triple_metrics import TripleMetrics
from .retrieval_metrics import RetrievalMetrics
from .text_metrics import TextMetrics
from utils.validation import warn_fallback_mode, QualityValidator


class OptimizedMetricsCalculator:
    """
    Optimized metrics calculator that processes all LLM operations once
    and shares results across multiple metrics to minimize API calls.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.llm_processor = OptimizedLLMProcessor(logger)
        self.validator = QualityValidator(logger)
        
        # Initialize metric classes with shared processor
        self.hallucination_metrics = HallucinationMetrics(logger, self.llm_processor)
        self.semantic_metrics = SemanticMetrics(logger, self.llm_processor)
        self.triple_metrics = TripleMetrics(logger, self.llm_processor)
        self.retrieval_metrics = RetrievalMetrics(logger, self.llm_processor)
        self.text_metrics = TextMetrics(logger)  # Text metrics don't need LLM
    
    async def compute_all_metrics(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        evidence: List[str],
        evidence_triples: List[str],
        contexts: List[str],
        llm_adapter: Any,
        embeddings_adapter: Optional[Any] = None,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Compute all metrics with optimized LLM usage.
        
        Args:
            question: The question being evaluated
            answer: The system-generated answer
            ground_truth: The ground truth answer
            evidence: List of evidence texts
            evidence_triples: List of evidence triples
            contexts: List of retrieved contexts
            llm_adapter: LLM adapter for making calls
            embeddings_adapter: Embeddings adapter (optional)
            max_retries: Maximum retries for LLM calls
            
        Returns:
            Dictionary containing all computed metrics
        """
        
        try:
            # Phase 1: Process all LLM operations once
            shared_results = await self.llm_processor.process_question_answer_pair(
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                evidence=evidence,
                evidence_triples=evidence_triples,
                contexts=contexts,
                llm_adapter=llm_adapter,
                max_retries=max_retries
            )
            
            # Phase 2: Compute all metrics using shared results
            metrics = {}
            
            # Compute metrics concurrently where possible
            hallucination_task = self._compute_hallucination_metrics(
                question, answer, evidence, evidence_triples, llm_adapter, max_retries, shared_results
            )
            
            semantic_task = self._compute_semantic_metrics(
                question, answer, ground_truth, llm_adapter, embeddings_adapter, max_retries, shared_results
            )
            
            triple_task = self._compute_triple_metrics(
                question, answer, evidence_triples, llm_adapter, shared_results
            )
            
            retrieval_task = self._compute_retrieval_metrics(
                question, answer, contexts, llm_adapter, max_retries, shared_results
            )
            
            text_task = self._compute_text_metrics(answer, ground_truth)
            
            # Wait for all metric computations
            results = await asyncio.gather(
                hallucination_task,
                semantic_task,
                triple_task,
                retrieval_task,
                text_task,
                return_exceptions=True
            )
            
            # Merge all metric results
            for result in results:
                if isinstance(result, dict):
                    metrics.update(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Metric computation failed: {result}")
            
            # Add processing statistics
            metrics['processing_stats'] = {
                'llm_calls_optimized': True,
                'shared_results_used': True,
                'cache_stats': self.llm_processor.get_cache_stats()
            }
            
            # Validate results
            expected_metrics = [
                'evidence_support_percentage', 'accuracy_percentage',
                'factual_accuracy_grade', 'semantic_similarity_percentage',
                'information_coverage_percentage', 'context_relevance_percentage',
                'answer_support_percentage', 'exact_match_score', 'f1_score'
            ]
            
            validation_passed = self.validator.validate_metrics_results(metrics, expected_metrics)
            if not validation_passed:
                self.logger.warning("Metrics validation failed - check validation summary")
                
            validation_summary = self.validator.get_validation_summary()
            metrics['validation_summary'] = validation_summary
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Optimized metrics calculation failed: {e}")
            # Use validation utility for consistent fallback warnings
            warn_fallback_mode(
                "OptimizedMetricsCalculator", 
                f"Optimization pipeline failed: {str(e)}"
            )
            # Fallback to individual metric computation without optimization
            return await self._fallback_compute_metrics(
                question, answer, ground_truth, evidence, evidence_triples, 
                contexts, llm_adapter, embeddings_adapter, max_retries
            )
    
    async def _compute_hallucination_metrics(
        self,
        question: str,
        answer: str,
        evidence: List[str],
        evidence_triples: List[str],
        llm_adapter: Any,
        max_retries: int,
        shared_results: SharedLLMResults
    ) -> Dict[str, float]:
        """Compute hallucination metrics using shared results."""
        
        metrics = {}
        
        try:
            # Evidence support percentage (compute first as it's needed for accuracy)
            metrics['evidence_support_percentage'] = await self.hallucination_metrics.compute_evidence_support_percentage(
                question, answer, evidence, llm_adapter, max_retries, shared_results
            )
            
            # Accuracy percentage (faithfulness-based) - use evidence support as faithfulness
            # Convert from percentage (0-100) back to ratio (0-1) for the method
            faithfulness = metrics['evidence_support_percentage'] / 100.0 if not np.isnan(metrics['evidence_support_percentage']) else None
            metrics['accuracy_percentage'] = self.hallucination_metrics.compute_accuracy_percentage(faithfulness)
            
            # Unsupported claims count
            metrics['unsupported_claims_count'] = await self.hallucination_metrics.compute_unsupported_claims_count(
                question, answer, evidence_triples, llm_adapter, shared_results
            )
            
            # Missing claims count
            metrics['missing_claims_count'] = await self.hallucination_metrics.compute_missing_claims_count(
                question, answer, evidence_triples, llm_adapter, shared_results
            )
            
        except Exception as e:
            self.logger.error(f"Hallucination metrics computation failed: {e}")
            # Set NaN values for failed metrics
            metrics.update({
                'accuracy_percentage': np.nan,
                'evidence_support_percentage': np.nan,
                'unsupported_claims_count': np.nan,
                'missing_claims_count': np.nan
            })
        
        return metrics
    
    async def _compute_semantic_metrics(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        llm_adapter: Any,
        embeddings_adapter: Optional[Any],
        max_retries: int,
        shared_results: SharedLLMResults
    ) -> Dict[str, float]:
        """Compute semantic metrics using shared results."""
        
        metrics = {}
        
        try:
            # Factual accuracy grade (A/B/C/D/E letter grade)
            metrics['factual_accuracy_grade'] = await self.semantic_metrics.compute_factual_accuracy_grade(
                question, answer, ground_truth, llm_adapter, shared_results
            )
            
            # Semantic similarity percentage
            if embeddings_adapter:
                metrics['semantic_similarity_percentage'] = await self.semantic_metrics.compute_semantic_similarity_percentage(
                    answer, ground_truth, embeddings_adapter
                )
            else:
                metrics['semantic_similarity_percentage'] = np.nan  # Cannot compute without embeddings adapter
            
            # Information coverage percentage
            metrics['information_coverage_percentage'] = await self.semantic_metrics.compute_information_coverage_percentage(
                question, ground_truth, answer, llm_adapter, max_retries, shared_results
            )
            
        except Exception as e:
            self.logger.error(f"Semantic metrics computation failed: {e}")
            # Set error/NaN values for failed metrics
            metrics.update({
                'factual_accuracy_grade': "ERROR",
                'semantic_similarity_percentage': np.nan,
                'information_coverage_percentage': np.nan
            })
        
        return metrics
    
    async def _compute_triple_metrics(
        self,
        question: str,
        answer: str,
        evidence_triples: List[str],
        llm_adapter: Any,
        shared_results: SharedLLMResults
    ) -> Dict[str, Any]:
        """Compute triple-based metrics using shared results."""
        
        metrics = {}
        
        try:
            # Correct facts count
            metrics['correct_facts_count'] = await self.triple_metrics.compute_correct_facts_count(
                question, answer, evidence_triples, llm_adapter, shared_results
            )
            
            # Incorrect facts count
            metrics['incorrect_facts_count'] = await self.triple_metrics.compute_incorrect_facts_count(
                question, answer, evidence_triples, llm_adapter, shared_results
            )
            
            # Fact extraction accuracy
            metrics['fact_extraction_accuracy'] = await self.triple_metrics.compute_fact_extraction_accuracy(
                question, answer, evidence_triples, llm_adapter, shared_results
            )
            
        except Exception as e:
            self.logger.error(f"Triple metrics computation failed: {e}")
            # Set NaN values for failed metrics
            metrics.update({
                'correct_facts_count': np.nan,
                'incorrect_facts_count': np.nan,
                'fact_extraction_accuracy': np.nan
            })
        
        return metrics
    
    async def _compute_retrieval_metrics(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        llm_adapter: Any,
        max_retries: int,
        shared_results: SharedLLMResults
    ) -> Dict[str, float]:
        """Compute retrieval metrics using shared results."""
        
        metrics = {}
        
        try:
            # Context relevance percentage
            metrics['context_relevance_percentage'] = await self.retrieval_metrics.compute_context_relevance_percentage(
                question, contexts, llm_adapter, max_retries, shared_results
            )
            
            # Answer support percentage
            metrics['answer_support_percentage'] = await self.retrieval_metrics.compute_answer_support_percentage(
                question, answer, contexts, llm_adapter, max_retries, shared_results
            )
            
        except Exception as e:
            self.logger.error(f"Retrieval metrics computation failed: {e}")
            # Set NaN values for failed metrics
            metrics.update({
                'context_relevance_percentage': np.nan,
                'answer_support_percentage': np.nan
            })
        
        return metrics
    
    async def _compute_text_metrics(self, answer: str, ground_truth: str) -> Dict[str, float]:
        """Compute text-based metrics (no LLM needed)."""
        
        try:
            return await self.text_metrics.compute_all_text_metrics(answer, ground_truth)
        except Exception as e:
            self.logger.error(f"Text metrics computation failed: {e}")
            return {
                'exact_match_score': np.nan,
                'f1_score': np.nan
            }
    
    async def _fallback_compute_metrics(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        evidence: List[str],
        evidence_triples: List[str],
        contexts: List[str],
        llm_adapter: Any,
        embeddings_adapter: Optional[Any],
        max_retries: int
    ) -> Dict[str, Any]:
        """Fallback to individual metric computation without optimization."""
        
        warn_fallback_mode(
            "OptimizedMetricsCalculator", 
            "Shared LLM processing failed, using individual metric computation"
        )
        
        self.logger.warning("Using fallback metric computation without optimization")
        
        metrics = {}
        
        # Initialize metrics without shared processor
        hallucination_metrics = HallucinationMetrics(self.logger)
        semantic_metrics = SemanticMetrics(self.logger)
        triple_metrics = TripleMetrics(self.logger)
        retrieval_metrics = RetrievalMetrics(self.logger)
        
        try:
            # Compute evidence support percentage first
            evidence_support = await hallucination_metrics.compute_evidence_support_percentage(
                question, answer, evidence, llm_adapter, max_retries, None
            )
            
            # Compute accuracy based on evidence support
            faithfulness = evidence_support / 100.0 if not np.isnan(evidence_support) else None
            metrics['accuracy_percentage'] = hallucination_metrics.compute_accuracy_percentage(faithfulness)
            metrics['evidence_support_percentage'] = evidence_support
            
            # Add warning about non-optimized computation
            metrics['processing_stats'] = {
                'llm_calls_optimized': False,
                'shared_results_used': False,
                'fallback_mode': True
            }
            
        except Exception as e:
            self.logger.error(f"Fallback metrics computation failed: {e}")
        
        return metrics
    
    def clear_cache(self):
        """Clear the LLM processor cache."""
        self.llm_processor.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.llm_processor.get_cache_stats()
