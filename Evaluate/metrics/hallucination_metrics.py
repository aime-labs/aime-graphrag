import json
import numpy as np
from typing import Any, Dict, List, Optional
import logging
from utils.error_handling import safe_metric_computation, parse_json_response, extract_first_number
from .optimized_llm_processor import OptimizedLLMProcessor, SharedLLMResults
from .geval_processor import RagasProcessor


class HallucinationMetrics:
    """Simple, interpretable hallucination detection metrics."""
    
    def __init__(self, logger: logging.Logger, llm_processor: Optional[OptimizedLLMProcessor] = None):
        self.logger = logger
        self.llm_processor = llm_processor or OptimizedLLMProcessor(logger)
        self.ragas_processor = RagasProcessor(logger)
    
    async def compute_evidence_support_percentage(self, question: str, answer: str, evidence: List[str],
                                                llm_adapter: Any, max_retries: int = 2, 
                                                shared_results: Optional[SharedLLMResults] = None,
                                                use_ragas: bool = True) -> float:
        """Compute what percentage of the answer is supported by evidence (0-100).
        
        Args:
            question: The question being answered
            answer: The system-generated answer
            evidence: List of evidence passages
            llm_adapter: LLM adapter for making calls
            max_retries: Maximum retry attempts
            shared_results: Optional shared results from optimized processor
            use_ragas: If True, use Ragas faithfulness evaluation (default: True)
        
        Returns:
            Evidence support/faithfulness score (0-100)
        """
        with safe_metric_computation('evidence_support_percentage', self.logger, fallback_value=np.nan):
            if not evidence or not any(e.strip() for e in evidence):
                return 0.0  # No evidence = legitimate zero support
            
            if use_ragas:
                # Use Ragas faithfulness evaluation for statement-level verification
                try:
                    ragas_result = await self.ragas_processor.evaluate_faithfulness_ragas(
                        question, answer, evidence, llm_adapter, max_retries
                    )
                    if not np.isnan(ragas_result.faithfulness_score):
                        return ragas_result.faithfulness_score
                    # If Ragas returns NaN, fallback to traditional method
                except Exception as e:
                    self.logger.warning(f"Ragas evidence support failed, falling back to traditional method: {e}")
                    # Fallback to traditional statement verification
                    pass
            
            # Use shared results if available
            if shared_results and shared_results.evidence_support_verdicts is not None:
                verdicts = shared_results.evidence_support_verdicts
            else:
                # Fallback to individual processing
                evidence_context = "\n".join(evidence)
                statements = await self._generate_statements(question, answer, llm_adapter, max_retries)
                
                if not statements:
                    # If no statements could be generated from answer
                    # Empty answer -> fully supported (100). If extraction failed, return NaN (computation failed)
                    return 100.0 if not answer.strip() else np.nan
                
                verdicts = await self._evaluate_evidence_support(statements, evidence_context, llm_adapter, max_retries)
            
            # Calculate support percentage
            if verdicts:
                supported = [v["verdict"] for v in verdicts]
                support_pct = (sum(supported) / len(supported)) * 100.0
                return support_pct
            
            # If we couldn't generate verdicts, try a simpler approach
            return await self._simple_evidence_check(answer, evidence, llm_adapter)
    
    async def compute_unsupported_claims_count(self, question: str, answer: str,
                                             evidence_tripples: List[str], llm_adapter: Any,
                                             shared_results: Optional[SharedLLMResults] = None) -> int:
        """Count how many claims in the answer are not supported by evidence."""
        with safe_metric_computation('unsupported_claims_count', self.logger, fallback_value=0):
            # If there is no evidence to compare against, skip expensive extraction and return 0
            if not evidence_tripples:
                return 0
            
            # Use shared results if available
            if shared_results and shared_results.answer_triples is not None and shared_results.ground_truth_triples is not None:
                answer_triples = shared_results.answer_triples
                gold_triples = shared_results.ground_truth_triples
            else:
                # Fallback to individual processing
                from .triple_metrics import TripleMetrics
                triple_metrics = TripleMetrics(self.logger)
                
                answer_triples = await triple_metrics._extract_system_triples_from_answer(answer, llm_adapter)
                gold_triples = triple_metrics._parse_evidence_tripples(evidence_tripples)
            
            if not answer_triples:
                return 0
            
            # Count unsupported claims
            unsupported = len(answer_triples - gold_triples)
            return unsupported
    
    async def compute_missing_claims_count(self, question: str, answer: str,
                                         evidence_tripples: List[str], llm_adapter: Any,
                                         shared_results: Optional[SharedLLMResults] = None) -> int:
        """Count how many important claims from evidence are missing in the answer."""
        with safe_metric_computation('missing_claims_count', self.logger, fallback_value=0):
            if not evidence_tripples:
                return 0
            
            # Use shared results if available
            if shared_results and shared_results.answer_triples is not None and shared_results.ground_truth_triples is not None:
                answer_triples = shared_results.answer_triples
                gold_triples = shared_results.ground_truth_triples
            else:
                # Fallback to individual processing
                from .triple_metrics import TripleMetrics
                triple_metrics = TripleMetrics(self.logger)
                
                answer_triples = await triple_metrics._extract_system_triples_from_answer(answer, llm_adapter)
                gold_triples = triple_metrics._parse_evidence_tripples(evidence_tripples)
            
            if not gold_triples:
                return 0
            
            # Count missing important claims
            missing = len(gold_triples - answer_triples)
            return missing
    
    async def _generate_statements(self, question: str, answer: str, llm_adapter: Any,
                                 max_retries: int) -> List[str]:
        """Generate atomic statements from answer."""
        prompt = f"""You are a precise statement extraction system. Break down the answer into atomic statements that are fully self-contained and understandable without context.

REQUIREMENTS:
- Each statement must be ATOMIC (one fact per statement)
- Replace ALL pronouns with the actual entity names
- Make each statement INDEPENDENT (fully understandable on its own)
- Preserve ALL factual information from the answer
- Use complete, grammatically correct sentences

EXAMPLE:
Question: "Who was Albert Einstein?"
Answer: "He was a German physicist known for relativity."
Output: ["Albert Einstein was a German physicist", "Albert Einstein is known for relativity"]

ACTUAL INPUT:
Question: "{question}"
Answer: "{answer}"

Respond ONLY with a JSON array of strings. Each string must be ONE atomic, self-contained statement.
Format: ["statement 1", "statement 2", "statement 3", ...]

Generated Statements:
"""
        
        for _ in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                return parse_json_response(
                    response.content,
                    self.logger,
                    "hallucination statement extraction",
                )
            except (json.JSONDecodeError, Exception):
                continue
        
        return []
    
    async def _evaluate_evidence_support(self, statements: List[str], evidence_context: str,
                                       llm_adapter: Any, max_retries: int) -> List[Dict]:
        """Evaluate which statements are supported by evidence."""
        prompt = f"""You are a harsh and critical evidence support evaluator. For each statement, determine if it can be DIRECTLY supported by the provided evidence.

BE STRICT:
- Mark as supported (verdict: 1) ONLY if the statement is explicitly present or clearly inferable from evidence
- Partial support is NOT enough - mark as unsupported (verdict: 0)
- Any missing details, inaccuracies, or ambiguities mean unsupported (verdict: 0)
- Require explicit evidence, not assumptions or implications

Evidence Context:
{evidence_context}

Statements to Evaluate:
{json.dumps(statements)}

Respond ONLY with a JSON array of objects. Each object must have:
{{
  "statement": "the exact statement being evaluated",
  "verdict": 1 (fully supported) or 0 (not supported or partially supported),
  "reason": "one-sentence explanation of why verdict was assigned"
}}

EXAMPLE RESPONSE FORMAT:
[
  {{"statement": "John is a computer science major", "verdict": 1, "reason": "Evidence explicitly states John studies Computer Science"}},
  {{"statement": "John works part-time", "verdict": 0, "reason": "No mention of employment status in evidence"}}
]

Your Response:
"""
        
        for _ in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                verdicts = parse_json_response(
                    response.content,
                    self.logger,
                    "hallucination verdicts",
                )
                return self._validate_verdicts(verdicts)
            except (json.JSONDecodeError, Exception):
                continue
        
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
    
    async def compute_claim_metrics(self, question: str, answer: str, evidence_tripples: List[str],
                                  llm_adapter: Any) -> Dict[str, float]:
        """Compute unsupported and missing claims count."""
        try:
            unsupported = await self.compute_unsupported_claims_count(question, answer, evidence_tripples, llm_adapter)
            missing = await self.compute_missing_claims_count(question, answer, evidence_tripples, llm_adapter)
            
            return {
                'unsupported_claims_count': float(unsupported),
                'missing_claims_count': float(missing)
            }
        except Exception as e:
            self.logger.error(f"Error computing claim metrics: {str(e)}")
            return {
                'unsupported_claims_count': 0.0,
                'missing_claims_count': 0.0
            }
    
    def compute_hallucination_rate(self, faithfulness: float) -> float:
        """Compute hallucination rate from faithfulness score."""
        return max(0.0, 100.0 - faithfulness)
    
    async def _simple_evidence_check(self, answer: str, evidence: List[str], llm_adapter: Any) -> float:
        """Perform a simple check if the answer is supported by evidence using classification."""
        try:
            evidence_str = "\n".join(evidence)
            prompt = f"""You are an evidence support evaluator. Classify how well the evidence supports the given answer.

CLASSIFICATION APPROACH:
First, classify the support level into ONE of these classes, then assign a score within that range:

CLASS A - FULLY SUPPORTED (Score: 85-100)
- All claims in the answer are explicitly verifiable from evidence
- No unsupported or contradictory information
- Example: Evidence: "Paris is France's capital", Answer: "The capital of France is Paris"

CLASS B - MOSTLY SUPPORTED (Score: 65-84)
- Main claims are supported by evidence
- Minor details may be unsupported but not contradictory
- Example: Evidence: "Einstein developed relativity", Answer: "Einstein developed relativity theory in 1915" (year not in evidence)

CLASS C - PARTIALLY SUPPORTED (Score: 40-64)
- Some claims are supported, others are not
- Mix of verified and unverified information
- Example: Evidence: "Python is a language", Answer: "Python is a fast compiled language" (partly true)

CLASS D - MINIMALLY SUPPORTED (Score: 20-39)
- Very few claims are supported by evidence
- Most information is unverified
- Example: Evidence: "Dogs are animals", Answer: "Dogs are blue flying mammals"

CLASS E - UNSUPPORTED (Score: 0-19)
- Answer contradicts evidence or is completely unrelated
- No meaningful support from evidence
- Example: Evidence: "Water freezes at 0°C", Answer: "Water freezes at 100°C"

INSTRUCTIONS:
1. Compare answer claims against evidence
2. Determine which class (A-E) best describes the support level
3. Assign a specific score within that class's range
4. Respond with: "CLASS: X, SCORE: YY"

Evidence: {evidence_str}

Answer: {answer}

Your Evaluation:"""
            
            response = await llm_adapter.ainvoke(prompt)
            response_text = response.content.strip()
            
            # Try to extract score from formatted response
            import re
            numbers = re.findall(r'SCORE:\s*(\d+)', response_text, re.IGNORECASE)
            if numbers:
                score = float(numbers[0])
                return min(100.0, max(0.0, score))
            
            # Fallback
            score = extract_first_number(
                response_text,
                self.logger,
                "simple evidence check",
            )
            if score is not None:
                return min(100.0, max(0.0, score))
            
            return np.nan  # Could not extract score - computation failed
        except Exception as e:
            self.logger.warning(f"Simple evidence check failed: {str(e)}")
            return np.nan  # Exception - computation failed
    
    async def compute_context_utilization(self, question: str, answer: str, 
                                          contexts: List[str], llm_adapter: Any,
                                          max_retries: int = 2) -> float:
        """
        Compute how well the answer utilizes the provided context.
        
        Measures what percentage of relevant context information is used in the answer.
        High utilization means the answer effectively incorporates context information.
        
        Args:
            question: The input question
            answer: The generated answer
            contexts: List of context passages (evidence)
            llm_adapter: LLM adapter for API calls
            max_retries: Maximum retry attempts
            
        Returns:
            Context utilization score (0-100)
        """
        with safe_metric_computation('context_utilization', self.logger, fallback_value=np.nan):
            if not answer or not answer.strip():
                return 0.0  # Empty answer = no utilization
            
            if not contexts or not any(c.strip() for c in contexts):
                return np.nan  # No context to utilize
            
            context_text = "\n\n".join([f"[Context {i+1}]:\n{ctx}" for i, ctx in enumerate(contexts)])
            
            prompt = f"""You are evaluating how well an answer utilizes provided context information.

**TASK:** Assess what percentage of relevant information from the contexts is used in the answer.

**EVALUATION CRITERIA:**
1. Identify key information in the contexts that is relevant to the question
2. Check how much of that relevant information appears in the answer
3. Consider both explicit mentions and paraphrased information

**SCORING GUIDE:**
- 90-100: Excellent utilization - answer uses most/all relevant context information
- 70-89: Good utilization - answer uses significant context information with minor omissions
- 50-69: Moderate utilization - answer uses some context but misses important details
- 30-49: Poor utilization - answer barely uses context, relies on other sources
- 0-29: Minimal utilization - answer ignores context almost entirely

**QUESTION:** {question}

**CONTEXTS:**
{context_text}

**ANSWER:** {answer}

**OUTPUT:**
Respond with ONLY a JSON object:
{{
  "utilization_score": 0-100,
  "key_info_used": ["list of key context info that was used"],
  "key_info_missed": ["list of relevant context info that was not used"],
  "reasoning": "brief explanation"
}}
"""
            
            for attempt in range(max_retries + 1):
                try:
                    response = await llm_adapter.ainvoke(prompt)
                    result = parse_json_response(
                        response.content, self.logger, "context utilization"
                    )
                    
                    if isinstance(result, dict) and "utilization_score" in result:
                        score = float(result["utilization_score"])
                        return max(0.0, min(100.0, score))
                        
                except Exception as e:
                    if attempt == max_retries:
                        self.logger.warning(f"Context utilization evaluation failed: {e}")
                    continue
            
            return np.nan