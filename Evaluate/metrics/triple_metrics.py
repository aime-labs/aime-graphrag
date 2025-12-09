import re
import json
import numpy as np
from typing import Any, Dict, List, Optional, Set
import logging
from utils.error_handling import safe_metric_computation, parse_json_response
from .optimized_llm_processor import OptimizedLLMProcessor, SharedLLMResults


class TripleMetrics:
    """Simple, interpretable triple-based evaluation metrics."""
    
    def __init__(self, logger: logging.Logger, llm_processor: Optional[OptimizedLLMProcessor] = None):
        self.logger = logger
        self.llm_processor = llm_processor or OptimizedLLMProcessor(logger)
    
    async def compute_correct_facts_count(self, question: str, answer: str, 
                                        ground_truth_triples: List[str], llm_adapter: Any,
                                        shared_results: Optional[SharedLLMResults] = None) -> int:
        """Count how many facts in the answer are correct."""
        with safe_metric_computation('correct_facts_count', self.logger, fallback_value=0):
            # Use shared results if available
            if shared_results and shared_results.answer_triples is not None and shared_results.ground_truth_triples is not None:
                answer_triples = shared_results.answer_triples
                gold_triples = shared_results.ground_truth_triples
            else:
                # Fallback to individual processing
                answer_triples = await self._extract_system_triples_from_answer(answer, llm_adapter)
                gold_triples = self._parse_evidence_tripples(ground_truth_triples)
            
            if not answer_triples:
                return 0
            
            # Count matching facts
            correct_count = len(answer_triples & gold_triples)
            return correct_count
    
    async def compute_incorrect_facts_count(self, question: str, answer: str,
                                          ground_truth_triples: List[str], llm_adapter: Any,
                                          shared_results: Optional[SharedLLMResults] = None) -> int:
        """Count how many facts in the answer are incorrect."""
        with safe_metric_computation('incorrect_facts_count', self.logger, fallback_value=0):
            # Use shared results if available
            if shared_results and shared_results.answer_triples is not None and shared_results.ground_truth_triples is not None:
                answer_triples = shared_results.answer_triples
                gold_triples = shared_results.ground_truth_triples
            else:
                # Fallback to individual processing
                answer_triples = await self._extract_system_triples_from_answer(answer, llm_adapter)
                gold_triples = self._parse_evidence_tripples(ground_truth_triples)
            
            if not answer_triples:
                return 0
            
            # Count non-matching facts
            incorrect_count = len(answer_triples - gold_triples)
            return incorrect_count
    
    async def compute_fact_extraction_accuracy(self, question: str, answer: str,
                                             ground_truth_triples: List[str], llm_adapter: Any,
                                             shared_results: Optional[SharedLLMResults] = None) -> float:
        """Compute percentage of facts that are extracted correctly (0-100)."""
        with safe_metric_computation('fact_extraction_accuracy', self.logger, fallback_value=np.nan):
            # Use shared results if available
            if shared_results and shared_results.answer_triples is not None and shared_results.ground_truth_triples is not None:
                answer_triples = shared_results.answer_triples
                gold_triples = shared_results.ground_truth_triples
            else:
                # Fallback to individual processing
                answer_triples = await self._extract_system_triples_from_answer(answer, llm_adapter)
                gold_triples = self._parse_evidence_tripples(ground_truth_triples)
            
            # If no answer triples but gold triples exist, return 0.0 (legitimate zero - failed to extract any facts)
            if not answer_triples and gold_triples:
                return 0.0
            
            # If no gold triples, cannot compute accuracy (missing reference data)
            # Return np.nan to clearly indicate missing ground truth (not a valid 0.0 or 100.0 score)
            if not gold_triples:
                return np.nan
            
            # Calculate accuracy percentage based on how many gold triples are found in answer
            correct_count = len(answer_triples & gold_triples)
            total_gold_facts = len(gold_triples)
            
            # Accuracy is based on recall: how many of the ground truth facts did we extract
            accuracy = (correct_count / total_gold_facts) * 100.0 if total_gold_facts > 0 else 0.0
            return accuracy
    
    async def _extract_system_triples_from_answer(self, answer: str, llm_adapter: Any) -> Set[str]:
        """Extract factual triples from answer using LLM."""
        prompt = f"""You are a comprehensive fact extraction system. Extract ALL factual statements from the answer and format them as simple, atomic triples.

REQUIREMENTS:
- Extract EVERY distinct fact mentioned in the answer
- Each triple should be atomic (one fact only)
- Format: simple subject-relation-object or subject-predicate statements
- Be thorough and exhaustive - don't miss any facts
- Include names, numbers, dates, relationships, and attributes
- Normalize entity names consistently

EXAMPLE:
Answer: "John Smith was born in New York and works at Google."
Output: ["John Smith was born in New York", "John Smith works at Google"]

Answer: {answer}

Respond ONLY with a JSON array of strings. Each string should be ONE atomic factual triple.
Format: ["triple 1", "triple 2", "triple 3", ...]

Extracted triples:
"""
        
        for _ in range(3):  # Max retries
            try:
                response = await llm_adapter.ainvoke(prompt)
                triples_list = parse_json_response(
                    response.content,
                    self.logger,
                    "triple extraction",
                )
                # Normalize triples to lowercase for comparison
                normalized_triples = {triple.lower().strip() for triple in triples_list if triple.strip()}
                return normalized_triples
            except (json.JSONDecodeError, Exception):
                continue
        
        return set()
    
    def _parse_evidence_tripples(self, evidence_triples: List[str]) -> Set[str]:
        """Parse evidence triples into normalized set."""
        if not evidence_triples:
            return set()
        
        normalized_triples = set()
        for triple in evidence_triples:
            if isinstance(triple, str) and triple.strip():
                # Normalize to lowercase for comparison
                normalized_triple = triple.lower().strip()
                normalized_triples.add(normalized_triple)
        
        return normalized_triples
    
    async def compute_triple_metrics_improved(self, question: str, answer: str, contexts: List[str],
                                            evidence_triples: List[str], llm_adapter: Any) -> Dict[str, float]:
        """Compute improved triple metrics that return individual values."""
        try:
            correct_facts = await self.compute_correct_facts_count(question, answer, evidence_triples, llm_adapter)
            incorrect_facts = await self.compute_incorrect_facts_count(question, answer, evidence_triples, llm_adapter)
            fact_accuracy = await self.compute_fact_extraction_accuracy(question, answer, evidence_triples, llm_adapter)
            
            return {
                'correct_facts_count': float(correct_facts),
                'incorrect_facts_count': float(incorrect_facts),
                'fact_extraction_accuracy': float(fact_accuracy)
            }
        except Exception as e:
            self.logger.error(f"Error computing triple metrics: {str(e)}")
            return {
                'correct_facts_count': 0.0,
                'incorrect_facts_count': 0.0,
                'fact_extraction_accuracy': 0.0
            }
