import json
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
from dataclasses import dataclass
from utils.error_handling import safe_metric_computation, parse_json_response


@dataclass
class SharedLLMResults:
    """Container for shared LLM processing results to avoid duplicate calls."""
    
    # Statement generation results
    answer_statements: Optional[List[str]] = None
    ground_truth_statements: Optional[List[str]] = None
    
    # Triple extraction results
    answer_triples: Optional[Set[str]] = None
    ground_truth_triples: Optional[Set[str]] = None
    
    # Fact extraction results
    ground_truth_facts: Optional[List[str]] = None
    
    # Evidence support verdicts
    evidence_support_verdicts: Optional[List[Dict]] = None
    
    # Context relevance scores
    context_relevance_scores: Optional[List[float]] = None
    
    # Factual accuracy evaluation
    factual_accuracy_evaluation: Optional[Dict] = None
    
    # Information coverage results
    information_coverage_results: Optional[List[Dict]] = None


class OptimizedLLMProcessor:
    """
    Optimized LLM processor that performs common LLM operations once 
    and shares results across multiple metrics to reduce API calls.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._cache: Dict[str, SharedLLMResults] = {}
    
    async def process_question_answer_pair(
        self, 
        question: str, 
        answer: str, 
        ground_truth: str, 
        evidence: List[str],
        evidence_triples: List[str],
        contexts: List[str],
        llm_adapter: Any,
        max_retries: int = 2
    ) -> SharedLLMResults:
        """
        Process a question-answer pair once and generate all shared results.
        
        Args:
            question: The question being evaluated
            answer: The system-generated answer
            ground_truth: The ground truth answer
            evidence: List of evidence texts
            evidence_triples: List of evidence triples
            contexts: List of retrieved contexts
            llm_adapter: LLM adapter for making calls
            max_retries: Maximum retries for LLM calls
            
        Returns:
            SharedLLMResults containing all processed results
        """
        
        # Create cache key
        cache_key = self._create_cache_key(question, answer, ground_truth, evidence, contexts)
        
        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Initialize results container
        results = SharedLLMResults()
        
        # Process all LLM operations concurrently where possible
        await self._process_all_operations(
            question, answer, ground_truth, evidence, evidence_triples, 
            contexts, llm_adapter, max_retries, results
        )
        
        # Cache results
        self._cache[cache_key] = results
        
        return results
    
    async def _process_all_operations(
        self,
        question: str,
        answer: str, 
        ground_truth: str,
        evidence: List[str],
        evidence_triples: List[str],
        contexts: List[str],
        llm_adapter: Any,
        max_retries: int,
        results: SharedLLMResults
    ):
        """Process all LLM operations, optimizing for batch processing where possible."""
        
        # Phase 1: Generate foundational extractions concurrently
        await asyncio.gather(
            self._generate_statements_batch(answer, ground_truth, llm_adapter, results),
            self._extract_triples_batch(answer, ground_truth, evidence_triples, llm_adapter, results),
            self._extract_facts_from_ground_truth(question, ground_truth, llm_adapter, max_retries, results),
            return_exceptions=True
        )
        
        # Phase 2: Process operations that depend on phase 1 results
        await asyncio.gather(
            self._evaluate_evidence_support(question, answer, evidence, llm_adapter, max_retries, results),
            self._evaluate_context_relevance(question, contexts, llm_adapter, max_retries, results),
            self._evaluate_factual_accuracy(question, llm_adapter, max_retries, results),
            self._evaluate_information_coverage(question, answer, llm_adapter, max_retries, results),
            return_exceptions=True
        )
    
    async def _generate_statements_batch(
        self, 
        answer: str, 
        ground_truth: str, 
        llm_adapter: Any, 
        results: SharedLLMResults
    ):
        """Generate statements from both answer and ground truth in a single batch call."""
        
        batch_prompt = f"""
Generate concise independent statements from the given texts that represent factual claims.
Respond ONLY with a JSON object containing two arrays: "answer_statements" and "ground_truth_statements".

### Answer Text:
{answer}

### Ground Truth Text:
{ground_truth}

### Expected Format:
{{
    "answer_statements": ["statement1", "statement2", ...],
    "ground_truth_statements": ["statement1", "statement2", ...]
}}

### Generated Statements:
"""
        
        try:
            response = await llm_adapter.ainvoke(batch_prompt)
            batch_results = parse_json_response(
                response.content,
                self.logger,
                "batch statement generation",
            )
            
            results.answer_statements = batch_results.get("answer_statements", [])
            results.ground_truth_statements = batch_results.get("ground_truth_statements", [])
            
        except (json.JSONDecodeError, Exception) as e:
            self.logger.warning(f"Batch statement generation failed: {e}")
            # Fallback to individual calls
            results.answer_statements = await self._generate_statements_single(answer, llm_adapter)
            results.ground_truth_statements = await self._generate_statements_single(ground_truth, llm_adapter)
    
    async def _extract_triples_batch(
        self,
        answer: str,
        ground_truth: str, 
        evidence_triples: List[str],
        llm_adapter: Any,
        results: SharedLLMResults
    ):
        """Extract triples from answer and parse ground truth triples."""
        
        # Extract answer triples
        results.answer_triples = await self._extract_system_triples_from_answer(answer, llm_adapter)
        
        # Parse ground truth triples
        results.ground_truth_triples = self._parse_evidence_triples(evidence_triples)
    
    async def _extract_facts_from_ground_truth(
        self,
        question: str,
        ground_truth: str,
        llm_adapter: Any,
        max_retries: int,
        results: SharedLLMResults
    ):
        """Extract factual claims from ground truth."""
        
        prompt = f"""
Extract factual claims from the reference text that are relevant to the question.
Respond ONLY with a JSON array of strings, each representing one factual claim.

Question: {question}
Reference: {ground_truth}

Extracted Facts:
"""
        
        for _ in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                results.ground_truth_facts = parse_json_response(
                    response.content,
                    self.logger,
                    "ground truth fact extraction",
                )
                break
            except (json.JSONDecodeError, Exception):
                continue
        
        if results.ground_truth_facts is None:
            results.ground_truth_facts = []
    
    async def _evaluate_evidence_support(
        self,
        question: str,
        answer: str,
        evidence: List[str],
        llm_adapter: Any,
        max_retries: int,
        results: SharedLLMResults
    ):
        """Evaluate which statements from answer are supported by evidence."""
        
        if not results.answer_statements or not evidence:
            results.evidence_support_verdicts = []
            return
        
        evidence_context = "\n".join(evidence)
        
        prompt = f"""
For each statement, determine if it can be directly supported by the provided evidence.
Respond ONLY with a JSON array of objects, each containing:
- "statement": the exact statement
- "verdict": 1 (supported by evidence) or 0 (not supported by evidence)
- "reason": brief explanation (1 sentence)

### Evidence Context
{evidence_context}

### Statements to Evaluate
{json.dumps(results.answer_statements)}

### Your Response:
"""
        
        for _ in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                verdicts = parse_json_response(
                    response.content,
                    self.logger,
                    "evidence support verdicts",
                )
                results.evidence_support_verdicts = self._validate_verdicts(verdicts)
                break
            except (json.JSONDecodeError, Exception):
                continue
        
        if results.evidence_support_verdicts is None:
            results.evidence_support_verdicts = []
    
    async def _evaluate_context_relevance(
        self,
        question: str,
        contexts: List[str],
        llm_adapter: Any,
        max_retries: int,
        results: SharedLLMResults
    ):
        """Evaluate relevance of each context to the question."""
        
        if not contexts:
            results.context_relevance_scores = []
            return
        
        # Batch evaluate all contexts
        batch_prompt = f"""
Rate the relevance of each context to the given question on a scale of 0-100.
Respond ONLY with a JSON array of numbers (one score per context).

Question: {question}

Contexts:
{json.dumps(contexts, indent=2)}

Relevance Scores (0-100):
"""
        
        for _ in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(batch_prompt)
                scores = parse_json_response(
                    response.content,
                    self.logger,
                    "context relevance scores",
                )
                results.context_relevance_scores = [float(s) for s in scores if isinstance(s, (int, float))]
                break
            except (json.JSONDecodeError, Exception):
                continue
        
        if results.context_relevance_scores is None:
            results.context_relevance_scores = [0.0] * len(contexts)
    
    async def _evaluate_factual_accuracy(
        self,
        question: str,
        llm_adapter: Any,
        max_retries: int,
        results: SharedLLMResults
    ):
        """Evaluate factual accuracy of answer statements against ground truth."""
        
        if not results.answer_statements or not results.ground_truth_statements:
            results.factual_accuracy_evaluation = {"evaluations": []}
            return
        
        prompt = f"""
Rate the factual accuracy of the answer statements compared to the ground truth.
For each answer statement, determine if it is:
- CORRECT: Factually accurate and supported by ground truth
- INCORRECT: Factually wrong or contradicts ground truth  
- PARTIALLY_CORRECT: Contains some truth but missing details or partially wrong

Respond with JSON containing a list of evaluations:
{{"evaluations": [{{"statement": "...", "rating": "CORRECT|INCORRECT|PARTIALLY_CORRECT", "reason": "..."}}]}}

Question: "{question}"
Answer Statements: {json.dumps(results.answer_statements)}
Ground Truth Statements: {json.dumps(results.ground_truth_statements)}
"""
        
        for _ in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                results.factual_accuracy_evaluation = parse_json_response(
                    response.content,
                    self.logger,
                    "factual accuracy evaluation",
                )
                break
            except (json.JSONDecodeError, Exception):
                continue
        
        if results.factual_accuracy_evaluation is None:
            results.factual_accuracy_evaluation = {"evaluations": []}
    
    async def _evaluate_information_coverage(
        self,
        question: str,
        answer: str,
        llm_adapter: Any,
        max_retries: int,
        results: SharedLLMResults
    ):
        """Check which facts from ground truth are covered in the answer."""
        
        if not results.ground_truth_facts:
            results.information_coverage_results = []
            return
        
        prompt = f"""
For each factual claim, determine if it is covered in the response.
Respond with a JSON array of objects, each containing:
- "fact": the factual claim
- "verdict": 1 if covered, 0 if not covered
- "reason": brief explanation

Question: {question}
Facts: {json.dumps(results.ground_truth_facts)}
Response: {answer}

Coverage Analysis:
"""
        
        for _ in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                results.information_coverage_results = parse_json_response(
                    response.content,
                    self.logger,
                    "information coverage",
                )
                break
            except (json.JSONDecodeError, Exception):
                continue
        
        if results.information_coverage_results is None:
            results.information_coverage_results = []
    
    async def _generate_statements_single(self, text: str, llm_adapter: Any) -> List[str]:
        """Generate statements from a single text (fallback method)."""
        
        prompt = f"""
Generate concise independent statements from the given text that represent factual claims.
Respond ONLY with a JSON array of strings. Do not include any other text.

Input Text: {text}

Generated Statements:
"""
        
        try:
            response = await llm_adapter.ainvoke(prompt)
            return parse_json_response(
                response.content,
                self.logger,
                "single statement extraction",
            )
        except (json.JSONDecodeError, Exception):
            return []
    
    async def _extract_system_triples_from_answer(self, answer: str, llm_adapter: Any) -> Set[str]:
        """Extract factual triples from answer using LLM."""
        
        prompt = f"""
Extract factual statements from the given answer and format them as simple triples (subject, relation, object).
Respond ONLY with a JSON array of strings, each representing one triple.

Example:
Answer: "John Smith was born in New York and works at Google."
Output: ["John Smith was born in New York", "John Smith works at Google"]

Answer: {answer}

Extracted triples:
"""
        
        for _ in range(3):  # Max retries
            try:
                response = await llm_adapter.ainvoke(prompt)
                triples_list = parse_json_response(
                    response.content,
                    self.logger,
                    "optimized triple extraction",
                )
                # Normalize triples to lowercase for comparison
                normalized_triples = {triple.lower().strip() for triple in triples_list if triple.strip()}
                return normalized_triples
            except (json.JSONDecodeError, Exception):
                continue
        
        return set()
    
    def _parse_evidence_triples(self, evidence_triples: List[str]) -> Set[str]:
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
    
    def _create_cache_key(
        self, 
        question: str, 
        answer: str, 
        ground_truth: str, 
        evidence: List[str], 
        contexts: List[str]
    ) -> str:
        """Create a cache key for the processed results."""
        
        # Create a hash-like key from the inputs
        import hashlib
        
        content = f"{question}|{answer}|{ground_truth}|{'|'.join(evidence)}|{'|'.join(contexts)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the cache to free memory."""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cached_entries": len(self._cache),
            "total_memory_usage_mb": sum(
                len(str(result.__dict__)) for result in self._cache.values()
            ) // (1024 * 1024)
        }
