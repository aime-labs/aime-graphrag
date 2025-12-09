import json
import numpy as np
from typing import Any, Dict, List, Optional
import logging
from utils.error_handling import safe_metric_computation, parse_json_response
from .optimized_llm_processor import OptimizedLLMProcessor, SharedLLMResults
from .geval_processor import GEvalProcessor
import asyncio


DEFAULT_CLASSIFICATION_PROMPT = (
    "You are a binary answer verifier. Compare the candidate answer to the reference answer for the given question.\n"
    "Label the candidate CORRECT only when it conveys the same factual meaning as the reference.\n"
    "Evaluation rules:\n"
    "- Treat the reference answer as authoritative.\n"
    "- Allow paraphrasing, reordered clauses, synonyms, or equivalent numeric units that preserve meaning.\n"
    "- Mark the answer WRONG if it contradicts the reference, omits an essential element, introduces incorrect facts, or answers a different question.\n"
    "- When unsure or when the reference facts are not satisfied, choose WRONG.\n"
    "Respond with a single word: CORRECT or WRONG.\n\n"
    "Question: {question}\n"
    "Reference Answer: {ground_truth}\n"
    "Candidate Answer: {answer}\n"
    "Classification:"
)


QUESTION_TYPE_CLASSIFICATION_PROMPTS: Dict[str, str] = {
    "Fact Retrieval": (
        "You are evaluating a Fact Retrieval answer. These questions expect a precise factual statement that matches the reference.\n"
        "Rules:\n"
        "1. Accept the candidate only if it communicates the same fact as the reference, allowing minor wording differences, synonyms, or equivalent measurements.\n"
        "2. If any required entity, value, qualifier, or relationship differs from the reference, mark it WRONG.\n"
        "3. Extra correct context is acceptable, but conflicting or alternative facts make the answer WRONG.\n"
        "4. Hedges, multiple options, or uncertainty signals should be treated as WRONG.\n"
        "Respond with a single token: CORRECT or WRONG.\n\n"
        "Question: {question}\n"
        "Reference Answer: {ground_truth}\n"
        "Candidate Answer: {answer}\n"
        "Classification:"
    ),
    "Complex Reasoning": (
        "You are evaluating a Complex Reasoning answer. These questions require combining multiple facts or steps to reach a conclusion that must align with the reference answer.\n"
        "Rules:\n"
        "1. Confirm that the final conclusion in the candidate exactly matches the reference conclusion.\n"
        "2. Ensure the candidate does not introduce contradictory or missing intermediate steps that would change the outcome.\n"
        "3. Partial alignment, incomplete chains, or logically inconsistent reasoning must be labelled WRONG.\n"
        "4. Only label CORRECT when the conclusion and essential supporting rationale agree with the reference.\n"
        "Respond with CORRECT or WRONG only.\n\n"
        "Question: {question}\n"
        "Reference Answer: {ground_truth}\n"
        "Candidate Answer: {answer}\n"
        "Classification:"
    ),
    "Contextual Summarize": (
        "You are evaluating a Contextual Summarize answer. The goal is to capture the same key points, emphasis, and factual claims found in the reference summary.\n"
        "Rules:\n"
        "1. Mark CORRECT only if the candidate includes all critical points from the reference without adding contradictory or fabricated details.\n"
        "2. Missing major ideas, misrepresenting tone, or altering factual claims should result in WRONG.\n"
        "3. Extra supporting detail is acceptable when it remains faithful to the reference content.\n"
        "Respond with CORRECT or WRONG only.\n\n"
        "Question: {question}\n"
        "Reference Answer: {ground_truth}\n"
        "Candidate Answer: {answer}\n"
        "Classification:"
    ),
    "Creative Generation": (
        "You are evaluating a Creative Generation answer. Even creative tasks have required attributes defined by the reference.\n"
        "Rules:\n"
        "1. Label the candidate CORRECT only if it satisfies every mandatory element shown in the reference answer (themes, constraints, characters, style notes, etc.).\n"
        "2. If the candidate diverges from any required attribute, contradicts the reference, or omits essential components, mark it WRONG.\n"
        "3. Tolerate stylistic or wording differences as long as the required creative constraints are met.\n"
        "Respond with CORRECT or WRONG.\n\n"
        "Question: {question}\n"
        "Reference Answer: {ground_truth}\n"
        "Candidate Answer: {answer}\n"
        "Classification:"
    ),
    "Retrieval": (
        "You are evaluating a Retrieval-style answer. The candidate must return the exact information or entity that the reference specifies.\n"
        "Rules:\n"
        "1. Mark CORRECT only when the candidate retrieves the same item(s) as the reference, including necessary qualifiers or identifiers.\n"
        "2. If the candidate provides different items, partial lists, or includes incorrect extras, mark it WRONG.\n"
        "3. Maintain strict alignment with the reference; near misses count as WRONG.\n"
        "Respond with CORRECT or WRONG.\n\n"
        "Question: {question}\n"
        "Reference Answer: {ground_truth}\n"
        "Candidate Answer: {answer}\n"
        "Classification:"
    ),
}


class SemanticMetrics:
    """Simple, interpretable semantic evaluation metrics."""
    
    def __init__(self, logger: logging.Logger, llm_processor: Optional[OptimizedLLMProcessor] = None):
        self.logger = logger
        self.llm_processor = llm_processor or OptimizedLLMProcessor(logger)
        self.geval_processor = GEvalProcessor(logger)
    
    async def compute_factual_accuracy_grade(self, question: str, answer: str, ground_truth: str,
                                                llm_adapter: Any, shared_results: Optional[SharedLLMResults] = None,
                                                use_geval: bool = True) -> str:
        """Compute how factually accurate the answer is compared to ground truth. 
        
        Args:
            question: The question being answered
            answer: The system-generated answer
            ground_truth: The reference answer
            llm_adapter: LLM adapter for making calls
            shared_results: Optional shared results from optimized processor
            use_geval: If True, use GEval multi-criteria evaluation (default: True)
        
        Returns:
            Factual accuracy grade as a letter (A, B, C, D, or E)
            - A: Excellent - completely accurate and comprehensive
            - B: Good - mostly correct with all key facts accurate
            - C: Adequate - captures main idea correctly but incomplete
            - D: Poor - significant factual errors or contradictions
            - E: Incorrect - entirely wrong or irrelevant
        """
        with safe_metric_computation('factual_accuracy_grade', self.logger, fallback_value="ERROR"):
            if use_geval:
                # Use GEval with Chain-of-Thought reasoning and multi-criteria scoring
                try:
                    geval_result = await self.geval_processor.evaluate_factual_accuracy_geval(
                        question, answer, ground_truth, llm_adapter
                    )
                    # Convert GEval score to classification
                    score = geval_result.final_score
                    if score >= 90:
                        return "A"
                    elif score >= 75:
                        return "B"
                    elif score >= 50:
                        return "C"
                    elif score >= 25:
                        return "D"
                    else:
                        return "E"
                except Exception as e:
                    self.logger.warning(f"GEval factual accuracy failed, falling back to simple method: {e}")
                    # Fallback to simple method if GEval fails
                    return await self._compute_simple_factual_accuracy(question, answer, ground_truth, llm_adapter)
            else:
                # Use simple classification-based method
                return await self._compute_simple_factual_accuracy(question, answer, ground_truth, llm_adapter)

    async def _compute_simple_factual_accuracy(self, question: str, answer: str, 
                                              ground_truth: str, llm_adapter: Any) -> str:
        """Compute factual accuracy with simple direct comparison using classification-based approach.
        
        Returns:
            Classification class (A, B, C, D, or E) as a string.
        """
        
        prompt = f"""You are a factual accuracy evaluator. Your task is to compare a candidate answer against the ground truth reference and classify it into one of five accuracy classes.

CLASSIFICATION APPROACH:
Classify the answer into ONE of these classes:

CLASS A - EXCELLENT
- Answer is completely accurate and comprehensive
- Matches ground truth exactly or with only trivial wording differences
- All key facts are present and correct
- Example: GT: "Paris is the capital of France", Answer: "The capital of France is Paris"

CLASS B - GOOD
- Answer is mostly correct with all key facts accurate
- May have minor omissions of non-critical details
- No factual errors present
- Example: GT: "Einstein developed relativity in 1915", Answer: "Einstein developed the theory of relativity" (missing year but factually correct)

CLASS C - ADEQUATE
- Answer captures the main idea correctly
- Missing some important details or context
- No major factual errors but incomplete
- Example: GT: "Python was created by Guido van Rossum in 1991", Answer: "Python is a programming language" (correct but very incomplete)

CLASS D - POOR
- Answer has significant factual errors or contradictions
- May have some correct elements mixed with wrong information
- Substantially differs from ground truth
- Example: GT: "The Earth orbits the Sun", Answer: "The Sun orbits the Earth"

CLASS E - INCORRECT
- Answer is entirely wrong or completely irrelevant
- Fabricated information or hallucinations
- No meaningful alignment with ground truth
- Example: GT: "Water boils at 100°C", Answer: "Water boils at 50°C"

INSTRUCTIONS:
1. Read both the ground truth and candidate answer carefully
2. Determine which class (A, B, C, D, or E) best describes the answer
3. Respond with ONLY the classification letter in this format: "CLASS: X"

Question: {question}

Ground Truth (Reference): {ground_truth}

Candidate Answer (To Evaluate): {answer}

Your Evaluation:"""
        
        try:
            response = await llm_adapter.ainvoke(prompt)
            response_text = response.content.strip().upper()
            
            # Extract classification class from response
            import re
            class_match = re.search(r'CLASS:\s*([A-E])', response_text, re.IGNORECASE)
            if class_match:
                return class_match.group(1).upper()
            
            # Fallback: try to find any single letter A-E in the response
            letter_match = re.search(r'\b([A-E])\b', response_text)
            if letter_match:
                return letter_match.group(1).upper()
            else:
                self.logger.warning(f"Could not parse classification from: {response_text}")
                return "UNKNOWN"  # Parse failed - return unknown class
                
        except Exception as e:
            self.logger.error(f"Error in factual accuracy classification: {str(e)}")
            return "ERROR"  # Exception - return error class
    
    async def compute_answer_classification(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        llm_adapter: Any,
        question_type: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Classify answer as correct, wrong, or don't know.
        Returns counts: correct_answers_count, wrong_answers_count, dont_know_answers_count
        """
        with safe_metric_computation('answer_classification', self.logger, fallback_value={'correct_answers_count': 0, 'wrong_answers_count': 0, 'dont_know_answers_count': 0}):
            # Check if answer indicates uncertainty/unknown
            uncertainty_phrases = [
                "i don't know", "i do not know", "i dont know", "unknown", "not sure", 
                "unclear", "cannot determine", "no information", "not found",
                "unable to answer", "insufficient information", "not available",
                "i'm not sure", "im not sure", "unsure", "not certain",
                "can't tell", "cannot tell", "no way to know", "impossible to determine",
                "no data", "no answer", "cannot say", "can't say"
            ]
            
            answer_lower = answer.lower().strip()
            is_uncertain = any(phrase in answer_lower for phrase in uncertainty_phrases)
            
            if is_uncertain:
                return {
                    'correct_answers_count': 0.0,
                    'wrong_answers_count': 0.0,
                    'dont_know_answers_count': 1.0
                }
            
            # Use LLM to classify as correct or wrong with a question-type aligned prompt
            prompt = self._build_answer_classification_prompt(
                question_type,
                question,
                ground_truth,
                answer
            )
            
            try:
                response = await llm_adapter.ainvoke(prompt)
                classification = response.content.strip().upper()
                
                if "CORRECT" in classification:
                    return {
                        'correct_answers_count': 1.0,
                        'wrong_answers_count': 0.0,
                        'dont_know_answers_count': 0.0
                    }
                else:
                    return {
                        'correct_answers_count': 0.0,
                        'wrong_answers_count': 1.0,
                        'dont_know_answers_count': 0.0
                    }
                    
            except Exception as e:
                self.logger.error(f"Error in answer classification: {str(e)}")
                return {
                    'correct_answers_count': 0.0,
                    'wrong_answers_count': 0.0,
                    'dont_know_answers_count': 0.0
                }

    def _build_answer_classification_prompt(
        self,
        question_type: Optional[str],
        question: str,
        ground_truth: str,
        answer: str
    ) -> str:
        """Build the judge prompt tailored to the question type."""
        key = (question_type or '').strip()
        template = QUESTION_TYPE_CLASSIFICATION_PROMPTS.get(key, DEFAULT_CLASSIFICATION_PROMPT)
        return template.format(question=question, ground_truth=ground_truth, answer=answer)
    
    async def compute_semantic_similarity_percentage(self, answer: str, ground_truth: str,
                                                   embeddings_adapter: Any) -> float:
        """Compute semantic similarity between answer and ground truth (0-100)."""
        with safe_metric_computation('semantic_similarity_percentage', self.logger, fallback_value=np.nan):
            # Calculate content similarity (0-100)
            similarity_score = await self._calculate_content_similarity(
                embeddings_adapter, answer, ground_truth
            )
            
            return min(100.0, max(0.0, similarity_score))
    
    async def compute_information_coverage_percentage(self, question: str, ground_truth: str, answer: str,
                                                    llm_adapter: Any, max_retries: int = 2,
                                                    shared_results: Optional[SharedLLMResults] = None,
                                                    use_geval: bool = True) -> float:
        """Compute what percentage of key information from ground truth is covered (0-100).
        
        Args:
            question: The question being answered
            ground_truth: The reference answer with all key information
            answer: The system-generated answer to evaluate
            llm_adapter: LLM adapter for making calls
            max_retries: Maximum retry attempts
            shared_results: Optional shared results from optimized processor
            use_geval: If True, use GEval multi-criteria evaluation (default: True)
        
        Returns:
            Information coverage score (0-100)
        """
        with safe_metric_computation('information_coverage_percentage', self.logger, fallback_value=np.nan):
            if use_geval:
                # Use GEval with multi-criteria coverage analysis
                try:
                    geval_result = await self.geval_processor.evaluate_information_coverage_geval(
                        question, ground_truth, answer, llm_adapter
                    )
                    return geval_result.final_score
                except Exception as e:
                    self.logger.warning(f"GEval information coverage failed, falling back to simple method: {e}")
                    # Fallback to fact-based method
                    pass
            
            # Use fact-based coverage method (original or fallback)
            # Use shared results if available
            if shared_results and shared_results.information_coverage_results is not None:
                coverage_results = shared_results.information_coverage_results
                gt_facts = shared_results.ground_truth_facts or []
            else:
                # Fallback to individual processing
                gt_facts = await self._extract_facts(question, ground_truth, llm_adapter, max_retries)
                
                if not gt_facts:
                    return 100.0  # Perfect score if no facts to cover
                
                coverage_results = await self._check_fact_coverage(
                    question, gt_facts, answer, llm_adapter, max_retries
                )
            
            # Calculate coverage percentage from numeric scores
            if not gt_facts:
                return 100.0  # No facts to cover = perfect coverage
            
            # Sum up all coverage scores and calculate average
            total_score = sum(result.get('score', 0) for result in coverage_results)
            return (total_score / len(gt_facts)) if gt_facts else np.nan
    
    async def _calculate_content_similarity(self, embeddings_adapter: Any,
                                          answer: str, ground_truth: str) -> float:
        """Compute content similarity as a percentage (0-100)."""
        try:
            a_embed, gt_embed = await asyncio.gather(
                embeddings_adapter.aembed_query(answer),
                embeddings_adapter.aembed_query(ground_truth)
            )
            
            # Convert to numpy arrays
            a_embed = np.array(a_embed)
            gt_embed = np.array(gt_embed)
            
            # Compute cosine similarity
            cosine_sim = np.dot(a_embed, gt_embed) / (
                np.linalg.norm(a_embed) * np.linalg.norm(gt_embed))
            
            # Convert to percentage (cosine similarity ranges from -1 to 1)
            similarity_percentage = ((cosine_sim + 1) / 2) * 100.0
            return max(0.0, min(100.0, similarity_percentage))
            
        except Exception:
            return np.nan  # Embeddings computation failed
    
    async def _extract_facts(self, question: str, reference: str, llm_adapter: Any,
                           max_retries: int) -> List[str]:
        """Extract factual claims from reference text."""
        prompt = f"""You are a comprehensive fact extraction system. Extract ALL verifiable factual claims from the reference text that directly relate to the question.

REQUIREMENTS:
- Break down the reference into atomic, independent claims
- Each claim should be complete and self-contained
- Include ALL relevant details (names, numbers, dates, relationships)
- Be thorough and exhaustive - don't miss any facts
- Each fact should be verifiable against the reference

Question: {question}
Reference: {reference}

Respond ONLY with a JSON array of strings. Each string must be ONE atomic factual claim.
Format: ["fact 1", "fact 2", "fact 3", ...]

Extracted Facts:
"""
        
        for _ in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                return parse_json_response(
                    response.content,
                    self.logger,
                    "semantic fact extraction",
                )
            except (json.JSONDecodeError, Exception):
                continue
        
        return []
    
    async def _check_fact_coverage(self, question: str, facts: List[str], response: str,
                                 llm_adapter: Any, max_retries: int) -> List[Dict]:
        """Check which facts are covered in the response using classification approach."""
        prompt = f"""You are a fact coverage evaluator. For each factual claim, classify whether it is covered in the response.

CLASSIFICATION APPROACH:
For each fact, assign it to ONE of these coverage classes:

CLASS A - FULLY COVERED (Score: 100)
- Fact is explicitly stated or clearly implied in the response
- All key details are present
- Information is accurate
- Example: Fact: "Paris is capital of France", Response: "France's capital is Paris" ✓

CLASS B - PARTIALLY COVERED (Score: 50)
- Main idea is present but missing important details
- Or contains the information with some inaccuracy
- Example: Fact: "Einstein developed relativity in 1915", Response: "Einstein worked on relativity" (missing year)

CLASS C - NOT COVERED (Score: 0)
- Fact is absent from the response
- Or fact is contradicted by the response
- Example: Fact: "Paris is capital of France", Response: "Lyon is a French city" ✗

INSTRUCTIONS:
1. For each fact, determine if it's fully covered (A), partially covered (B), or not covered (C)
2. Assign the corresponding score (100, 50, or 0)
3. Provide a brief reason

Question: {question}
Facts: {json.dumps(facts)}
Response: {response}

Respond ONLY with a JSON array of objects. Each object must have:
{{
  "fact": "the factual claim being evaluated",
  "score": 100, 50, or 0,
  "reason": "brief explanation (max 20 words)"
}}

Format: [{{"fact": "...", "score": 100, "reason": "..."}}, ...]

Coverage Analysis:
"""
        
        for _ in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                return parse_json_response(
                    response.content,
                    self.logger,
                    "semantic fact coverage",
                )
            except (json.JSONDecodeError, Exception):
                continue
        
        return []