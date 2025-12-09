import json
import numpy as np
from typing import Any, Dict, List, Optional
import logging
from utils.error_handling import safe_metric_computation, parse_json_response, extract_first_number
from .optimized_llm_processor import OptimizedLLMProcessor, SharedLLMResults
from .geval_processor import GEvalProcessor, RagasProcessor
from .ragas_metrics import truncate_contexts, MAX_CONTEXT_SIZE, MAX_SINGLE_CONTEXT_SIZE


class RetrievalMetrics:
    """Simple, interpretable retrieval evaluation metrics."""
    
    def __init__(self, logger: logging.Logger, llm_processor: Optional[OptimizedLLMProcessor] = None):
        self.logger = logger
        self.llm_processor = llm_processor or OptimizedLLMProcessor(logger)
        self.geval_processor = GEvalProcessor(logger)
        self.ragas_processor = RagasProcessor(logger)
    
    async def compute_context_relevance_percentage(self, question: str, contexts: List[str],
                                                 llm_adapter: Any, max_retries: int = 3,
                                                 shared_results: Optional[SharedLLMResults] = None,
                                                 use_geval: bool = True) -> float:
        """Compute how relevant the retrieved contexts are to the question (0-100).
        
        Args:
            question: The question being answered
            contexts: List of retrieved context passages
            llm_adapter: LLM adapter for making calls
            max_retries: Maximum retry attempts
            shared_results: Optional shared results from optimized processor
            use_geval: If True, use GEval multi-criteria evaluation (default: True)
        
        Returns:
            Context relevance score (0-100)
        """
        with safe_metric_computation('context_relevance_percentage', self.logger, fallback_value=np.nan):
            if not contexts:
                return 0.0  # No contexts = legitimate zero score (retrieval failed)
            
            if use_geval:
                # Use GEval with multi-criteria relevance assessment
                try:
                    geval_result = await self.geval_processor.evaluate_context_relevance_geval(
                        question, contexts, llm_adapter
                    )
                    return geval_result.final_score
                except Exception as e:
                    self.logger.warning(f"GEval context relevance failed, falling back to simple method: {e}")
                    # Fallback to per-context scoring
                    pass
            
            # Use shared results if available
            if shared_results and shared_results.context_relevance_scores is not None:
                relevance_scores = shared_results.context_relevance_scores
            else:
                # Fallback to individual processing
                relevance_scores = []
                for context in contexts:
                    score = await self._get_simple_relevance_rating(question, context, llm_adapter, max_retries)
                    relevance_scores.append(score)
            
            # Return average relevance as percentage
            return np.mean(relevance_scores) if relevance_scores else np.nan
    
    async def compute_answer_support_percentage(self, question: str, answer: str, contexts: List[str],
                                              llm_adapter: Any, max_retries: int = 2,
                                              shared_results: Optional[SharedLLMResults] = None,
                                              use_ragas: bool = True) -> float:
        """Compute what percentage of the answer is supported by the provided contexts (0-100).
        
        Args:
            question: The question being answered
            answer: The system-generated answer
            contexts: List of retrieved context passages
            llm_adapter: LLM adapter for making calls
            max_retries: Maximum retry attempts
            shared_results: Optional shared results from optimized processor
            use_ragas: If True, use Ragas faithfulness evaluation (default: True)
        
        Returns:
            Answer support/faithfulness score (0-100)
        """
        with safe_metric_computation('answer_support_percentage', self.logger, fallback_value=np.nan):
            # If no contexts provided, we cannot verify support
            if not contexts or not any(c.strip() for c in contexts):
                return 0.0  # No contexts = legitimate zero support
            
            if use_ragas:
                # Use Ragas faithfulness evaluation with statement-level verification
                try:
                    support_score, reasoning = await self.ragas_processor.evaluate_answer_support_ragas(
                        question, answer, contexts, llm_adapter
                    )
                    if not np.isnan(support_score):
                        return support_score
                    # If Ragas returns NaN, fallback to traditional method
                except Exception as e:
                    self.logger.warning(f"Ragas answer support failed, falling back to traditional method: {e}")
                    # Fallback to traditional statement verification
                    pass
            
            # Use shared results if available (evidence support verdicts can be reused for context support)
            if shared_results and shared_results.evidence_support_verdicts is not None:
                verdicts = shared_results.evidence_support_verdicts
            else:
                # Fallback to individual processing
                statements = await self._generate_statements(question, answer, llm_adapter, max_retries)
                
                if not statements:
                    return 100.0 if not answer.strip() else 0.0
                
                # Truncate contexts to prevent API overload
                truncated_contexts, was_truncated = truncate_contexts(contexts, logger=self.logger)
                context_str = "\n".join(truncated_contexts)
                verdicts = await self._evaluate_statements(statements, context_str, llm_adapter, max_retries)
            
            # Calculate support percentage
            if verdicts:
                supported = [v["verdict"] for v in verdicts]
                support_pct = (sum(supported) / len(supported)) * 100.0
                return support_pct
            
            # If we couldn't generate verdicts, try a simpler approach
            # Check if answer is contained in or similar to context
            return await self._simple_context_check(answer, contexts, llm_adapter)
    
    async def compute_context_completeness_percentage(self, question: str, contexts: List[str],
                                                    ground_truth: str, llm_adapter: Any) -> float:
        """Compute what percentage of needed information is present in retrieved contexts (0-100)."""
        with safe_metric_computation('context_completeness_percentage', self.logger, fallback_value=np.nan):
            if not contexts:
                return 0.0  # No contexts = legitimate zero completeness
            if not ground_truth:
                return np.nan  # Missing reference - cannot compute
            
            # Extract key information needed from ground truth
            prompt = f"""You are a thorough information analyst. Identify ALL key pieces of information needed to comprehensively answer this question based on the ground truth.

BE COMPREHENSIVE:
- Extract every distinct fact, concept, or detail mentioned in the ground truth
- Include names, numbers, dates, relationships, and explanations
- Break down complex information into atomic pieces
- Don't skip any relevant information

Question: {question}
Ground Truth: {ground_truth}

Respond ONLY with a JSON array of strings. Each string should be ONE key piece of information.
Format: ["info 1", "info 2", "info 3", ...]

Key information needed:
"""
            
            try:
                response = await llm_adapter.ainvoke(prompt)
                needed_info = parse_json_response(
                    response.content,
                    self.logger,
                    "context completeness needed info",
                )
                
                if not needed_info:
                    return 100.0
                
                # Check what percentage of needed info is in contexts
                context_str = "\n".join(contexts)
                found_count = 0
                
                for info in needed_info:
                    check_prompt = f"""You are a strict information verifier. Determine if the context contains the specific information requested.

BE HARSH:
- Answer "YES" ONLY if the context explicitly contains or clearly implies the information
- Partial information or vague mentions should be answered with "NO"
- Missing details or inaccuracies mean "NO"

Does the following context contain information about: "{info}"?

Context: {context_str}

Respond with ONLY "YES" or "NO" (no explanation needed).
"""
                    check_response = await llm_adapter.ainvoke(check_prompt)
                    if "YES" in check_response.content.upper():
                        found_count += 1
                
                return (found_count / len(needed_info)) * 100.0
                
            except Exception:
                return np.nan  # Computation failed
    

    
    async def _generate_statements(self, question: str, answer: str, llm_adapter: Any,
                                 max_retries: int) -> List[str]:
        """Break down answer into atomic statements."""
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
                    "answer support statements",
                )
            except (json.JSONDecodeError, Exception):
                continue
        
        return []
    
    async def _evaluate_statements(self, statements: List[str], context: str,
                                 llm_adapter: Any, max_retries: int) -> List[Dict]:
        """Evaluate which statements are supported by context."""
        prompt = f"""You are a harsh and critical statement verification judge. Evaluate if each statement can be DIRECTLY inferred from the context.

BE STRICT:
- Mark as supported (verdict: 1) ONLY if the statement is explicitly present or clearly inferable from the context
- Partial support is NOT enough - mark as unsupported (verdict: 0)
- Any missing details, inaccuracies, or ambiguities mean unsupported (verdict: 0)
- Require explicit evidence, not assumptions

Context:
{context}

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
  {{"statement": "John is a computer science major", "verdict": 1, "reason": "Context explicitly states John studies Computer Science"}},
  {{"statement": "John works part-time", "verdict": 0, "reason": "No mention of employment status in context"}}
]

Your Response:
"""
        
        for _ in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                verdicts = parse_json_response(
                    response.content,
                    self.logger,
                    "answer support verdicts",
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
    
    async def _get_simple_relevance_rating(self, question: str, context: str, llm_adapter: Any,
                                          max_retries: int) -> float:
        """Get simple relevance rating as percentage using classification-based approach."""
        prompt = f"""You are a context relevance evaluator. Classify how relevant this context is for answering the question.

CLASSIFICATION APPROACH:
First, classify the context into ONE of these relevance classes, then assign a score within that range:

CLASS A - HIGHLY RELEVANT (Score: 85-100)
- Context directly and completely answers the question
- All necessary information is present
- Minimal irrelevant content
- Example: Q: "What is the capital of France?", Context: "Paris is the capital and largest city of France"

CLASS B - RELEVANT (Score: 65-84)
- Context provides most information needed
- May have minor gaps or some tangential content
- Clear connection to the question
- Example: Q: "What is the capital of France?", Context: "France is a European country. Its main city is Paris, which serves as the political center"

CLASS C - PARTIALLY RELEVANT (Score: 40-64)
- Context has some useful information
- Significant gaps or mostly tangential content
- Indirect connection to the question
- Example: Q: "What is the capital of France?", Context: "France has many beautiful cities including Lyon, Marseille, and others"

CLASS D - MINIMALLY RELEVANT (Score: 20-39)
- Context barely relates to the question
- Mostly tangential or off-topic
- Very little useful information
- Example: Q: "What is the capital of France?", Context: "European cities are known for their architecture"

CLASS E - IRRELEVANT (Score: 0-19)
- Context is completely unrelated to the question
- No useful information present
- Example: Q: "What is the capital of France?", Context: "Python is a programming language"

INSTRUCTIONS:
1. Determine which class (A-E) best describes the context's relevance
2. Assign a specific score within that class's range
3. Respond with: "CLASS: X, SCORE: YY"

Question: {question}

Context: {context}

Your Evaluation:"""
        
        for _ in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                response_text = response.content.strip()
                
                # Try to extract score from formatted response
                import re
                numbers = re.findall(r'SCORE:\s*(\d+)', response_text, re.IGNORECASE)
                if numbers:
                    score = float(numbers[0])
                    return max(0.0, min(100.0, score))
                
                # Fallback: try to find any number
                numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response_text)
                if numbers:
                    score = float(numbers[0])
                    return max(0.0, min(100.0, score))  # Clamp to 0-100
                return np.nan  # Could not parse response
            except Exception:
                continue
        
        return np.nan  # All retries failed - computation failed
    
    async def _simple_context_check(self, answer: str, contexts: List[str], llm_adapter: Any) -> float:
        """Perform a simple check if the answer is supported by contexts using classification."""
        try:
            context_str = "\n".join(contexts)
            prompt = f"""You are an answer support evaluator. Classify how well the context supports the given answer.

CLASSIFICATION APPROACH:
First, classify the support level into ONE of these classes, then assign a score within that range:

CLASS A - FULLY SUPPORTED (Score: 85-100)
- All claims in the answer are explicitly verifiable from context
- No unsupported or contradictory information
- Example: Context: "Paris is France's capital", Answer: "The capital of France is Paris"

CLASS B - MOSTLY SUPPORTED (Score: 65-84)
- Main claims are supported by context
- Minor details may be unsupported but not contradictory
- Example: Context: "Einstein developed relativity", Answer: "Einstein developed the theory of relativity in 1915" (year not in context but not wrong)

CLASS C - PARTIALLY SUPPORTED (Score: 40-64)
- Some claims are supported, others are not
- Mix of verified and unverified information
- Example: Context: "Python is a language", Answer: "Python is a fast compiled language" (first part supported, second not)

CLASS D - MINIMALLY SUPPORTED (Score: 20-39)
- Very few claims are supported by context
- Most information is unverified
- Example: Context: "Dogs are animals", Answer: "Dogs are blue flying mammals" (only "dogs" and "animals" match)

CLASS E - UNSUPPORTED (Score: 0-19)
- Answer contradicts context or is completely unrelated
- No meaningful support from context
- Example: Context: "Water freezes at 0°C", Answer: "Water freezes at 100°C"

INSTRUCTIONS:
1. Compare answer claims against context
2. Determine which class (A-E) best describes the support level
3. Assign a specific score within that class's range
4. Respond with: "CLASS: X, SCORE: YY"

Context: {context_str}

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
                "simple context check",
            )
            if score is not None:
                return min(100.0, max(0.0, score))
            
            return np.nan  # Could not extract score - computation failed
        except Exception as e:
            self.logger.warning(f"Simple context check failed: {str(e)}")
            return np.nan  # Exception - computation failed

    # Alias methods for backward compatibility
    async def compute_context_relevance(self, question: str, contexts: List[str], llm_adapter: Any, **kwargs) -> float:
        """Alias for compute_context_relevance_percentage."""
        return await self.compute_context_relevance_percentage(question, contexts, llm_adapter, **kwargs)

    async def compute_context_completeness(self, question: str, contexts: List[str], ground_truth: str, llm_adapter: Any, **kwargs) -> float:
        """Alias for compute_context_completeness_percentage."""
        return await self.compute_context_completeness_percentage(question, contexts, ground_truth, llm_adapter, **kwargs)

    async def compute_faithfulness(self, question: str, answer: str, contexts: List[str], 
                                   llm_adapter: Any, max_retries: int = 2) -> float:
        """
        Compute faithfulness score - how well the answer is grounded in the contexts.
        
        This delegates to RAGAS faithfulness implementation which:
        1. Extracts claims from the answer
        2. Verifies each claim against the contexts
        3. Returns percentage of supported claims
        
        Args:
            question: The input question
            answer: The generated answer
            contexts: List of retrieved context passages
            llm_adapter: LLM adapter for API calls
            max_retries: Maximum retry attempts
            
        Returns:
            Faithfulness score (0-100)
        """
        # Import here to avoid circular imports
        from .ragas_metrics import RagasMetrics
        
        with safe_metric_computation('faithfulness', self.logger, fallback_value=np.nan):
            if not answer or not answer.strip():
                return 100.0  # Empty answer is trivially faithful
            
            if not contexts or not any(c.strip() for c in contexts):
                return 0.0  # No context = cannot verify faithfulness
            
            ragas = RagasMetrics(self.logger)
            score, _ = await ragas.compute_faithfulness(
                question, answer, contexts, llm_adapter, max_retries
            )
            return score
    
    async def compute_context_quality(self, contexts: List[str], llm_adapter: Any,
                                      max_retries: int = 2) -> float:
        """
        Compute overall quality score for retrieved contexts.
        
        Evaluates contexts based on:
        - Information density
        - Clarity and coherence
        - Relevance signals
        
        Args:
            contexts: List of retrieved context passages
            llm_adapter: LLM adapter for API calls
            max_retries: Maximum retry attempts
            
        Returns:
            Context quality score (0-100)
        """
        with safe_metric_computation('context_quality', self.logger, fallback_value=np.nan):
            if not contexts or not any(c.strip() for c in contexts):
                return 0.0
            
            context_text = "\n\n".join([f"[Context {i+1}]:\n{ctx}" for i, ctx in enumerate(contexts)])
            
            prompt = f"""You are evaluating the quality of retrieved text passages.
Rate the overall quality of these contexts on a scale of 0-100.

**QUALITY CRITERIA:**
1. Information Density (40%): How much useful, specific information is present?
2. Clarity (30%): Is the text clear, well-organized, and easy to understand?
3. Coherence (30%): Are the contexts internally consistent and logically structured?

**SCORING GUIDE:**
- 90-100: Exceptional quality - dense information, crystal clear, perfectly coherent
- 70-89: Good quality - useful information, mostly clear, good coherence
- 50-69: Adequate quality - some useful info, reasonably clear
- 30-49: Poor quality - sparse info, unclear, or incoherent
- 0-29: Very poor - little useful information, confusing, or contradictory

**CONTEXTS TO EVALUATE:**
{context_text}

**OUTPUT:**
Respond with ONLY a JSON object:
{{
  "quality_score": 0-100,
  "reasoning": "brief explanation"
}}
"""
            
            for attempt in range(max_retries + 1):
                try:
                    response = await llm_adapter.ainvoke(prompt)
                    result = parse_json_response(
                        response.content, self.logger, "context quality"
                    )
                    
                    if isinstance(result, dict) and "quality_score" in result:
                        score = float(result["quality_score"])
                        return max(0.0, min(100.0, score))
                        
                except Exception as e:
                    if attempt == max_retries:
                        self.logger.warning(f"Context quality evaluation failed: {e}")
                    continue
            
            return np.nan
    
    async def compute_information_retrieval_score(self, question: str, contexts: List[str],
                                                  ground_truth: str, embeddings_adapter: Any) -> float:
        """
        Compute information retrieval score using embeddings.
        
        Measures how well the retrieved contexts match the expected answer
        using semantic similarity.
        
        Args:
            question: The input question
            contexts: List of retrieved context passages
            ground_truth: The expected/reference answer
            embeddings_adapter: Embeddings adapter for computing similarity
            
        Returns:
            Information retrieval score (0-100)
        """
        import asyncio
        
        with safe_metric_computation('information_retrieval_score', self.logger, fallback_value=np.nan):
            if not contexts or not ground_truth:
                return 0.0
            
            try:
                # Get embedding for ground truth
                gt_embedding = await embeddings_adapter.aembed_query(ground_truth)
                gt_embedding = np.array(gt_embedding)
                
                # Get embeddings for all contexts and compute similarities
                similarities = []
                for ctx in contexts:
                    if ctx.strip():
                        ctx_embedding = await embeddings_adapter.aembed_query(ctx)
                        ctx_embedding = np.array(ctx_embedding)
                        
                        # Cosine similarity
                        cosine_sim = np.dot(gt_embedding, ctx_embedding) / (
                            np.linalg.norm(gt_embedding) * np.linalg.norm(ctx_embedding)
                        )
                        # Normalize from [-1, 1] to [0, 1]
                        normalized_sim = (cosine_sim + 1) / 2
                        similarities.append(normalized_sim)
                
                if not similarities:
                    return 0.0
                
                # Return max similarity (best matching context) as percentage
                return float(max(similarities)) * 100.0
                
            except Exception as e:
                self.logger.warning(f"Information retrieval score computation failed: {e}")
                return np.nan