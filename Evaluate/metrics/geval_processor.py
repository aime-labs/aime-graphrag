"""
GEval and Ragas-style LLM-as-Judge Processor

Implements advanced evaluation patterns with Chain-of-Thought reasoning,
multi-criteria scoring, and detailed explanations for improved accuracy
and transparency in GraphRAG/RAG/LLM evaluation.

References:
- GEval: NLG Evaluation using GPT-4 with Better Human Alignment (Liu et al., 2023)
- Ragas: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from utils.error_handling import safe_metric_computation, parse_json_response


@dataclass
class GEvalResult:
    """Container for GEval scoring results with reasoning traces."""
    
    # Individual criterion scores
    criterion_scores: Dict[str, float]
    criterion_reasoning: Dict[str, str]
    
    # Final weighted score
    final_score: float
    overall_reasoning: str
    
    # Metadata
    evaluation_type: str
    confidence: Optional[float] = None


@dataclass
class RagasResult:
    """Container for Ragas-style evaluation results."""
    
    # Faithfulness components
    statement_scores: Optional[List[Dict[str, Any]]] = None
    faithfulness_score: Optional[float] = None
    
    # Answer relevance components  
    question_generation: Optional[List[str]] = None
    relevance_score: Optional[float] = None
    
    # Context relevance components
    context_scores: Optional[List[Dict[str, Any]]] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    
    # Overall reasoning
    overall_reasoning: str = ""


class GEvalProcessor:
    """
    Implements GEval pattern for LLM-based evaluation with:
    - Chain-of-Thought (CoT) reasoning
    - Multi-criteria decomposition
    - Weighted scoring with explicit rubrics
    - Detailed explanation generation
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    async def evaluate_factual_accuracy_geval(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        llm_adapter: Any,
        max_retries: int = 2
    ) -> GEvalResult:
        """
        Evaluate factual accuracy using GEval multi-criteria approach.
        
        Criteria:
        1. Correctness (50%): Are the facts accurate?
        2. Completeness (30%): Is all important information included?
        3. Consistency (20%): Is the answer internally consistent?
        """
        
        prompt = f"""You are an expert factual accuracy evaluator. Use a systematic, step-by-step approach to evaluate the answer against the ground truth.

**EVALUATION TASK**: Assess factual accuracy of the candidate answer compared to the reference answer.

**CHAIN-OF-THOUGHT EVALUATION**:

Let's think through this step by step:

**Step 1: Correctness Analysis (Weight: 50%)**
- Task: Evaluate if all factual claims in the answer are accurate
- Consider: Are there any factually incorrect statements? Do facts align with ground truth?
- Scoring Rubric:
  * 90-100: All facts are completely accurate, no errors
  * 70-89: Minor inaccuracies that don't change core meaning
  * 50-69: Some notable factual errors but main points correct
  * 30-49: Multiple significant factual errors
  * 0-29: Predominantly incorrect or fabricated facts

**Step 2: Completeness Analysis (Weight: 30%)**
- Task: Evaluate if all essential information from ground truth is present
- Consider: Are key facts missing? Is depth of information adequate?
- Scoring Rubric:
  * 90-100: All essential information present with appropriate detail
  * 70-89: Most key information present, minor details missing
  * 50-69: Core information present but lacking important details
  * 30-49: Significant gaps in essential information
  * 0-29: Most essential information missing

**Step 3: Consistency Analysis (Weight: 20%)**
- Task: Evaluate internal logical consistency of the answer
- Consider: Are there contradictions? Does information fit together logically?
- Scoring Rubric:
  * 90-100: Completely consistent, no contradictions
  * 70-89: Mostly consistent with minor ambiguities
  * 50-69: Some inconsistencies but overall coherent
  * 30-49: Notable contradictions or logical gaps
  * 0-29: Highly inconsistent or contradictory

**INPUT DATA**:
Question: {question}
Ground Truth: {ground_truth}
Candidate Answer: {answer}

**OUTPUT FORMAT** (JSON only):
{{
  "step1_correctness": {{
    "score": 0-100,
    "reasoning": "Detailed explanation of correctness assessment"
  }},
  "step2_completeness": {{
    "score": 0-100,
    "reasoning": "Detailed explanation of completeness assessment"
  }},
  "step3_consistency": {{
    "score": 0-100,
    "reasoning": "Detailed explanation of consistency assessment"
  }},
  "final_score": "weighted average: (correctness*0.5 + completeness*0.3 + consistency*0.2)",
  "overall_reasoning": "Summary of evaluation with key findings",
  "confidence": 0-100
}}

**YOUR EVALUATION**:"""
        
        for attempt in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                result = parse_json_response(
                    response.content,
                    self.logger,
                    "geval factual accuracy"
                )
                
                # Extract scores and reasoning
                criterion_scores = {
                    "correctness": float(result.get("step1_correctness", {}).get("score", 0)),
                    "completeness": float(result.get("step2_completeness", {}).get("score", 0)),
                    "consistency": float(result.get("step3_consistency", {}).get("score", 0))
                }
                
                criterion_reasoning = {
                    "correctness": result.get("step1_correctness", {}).get("reasoning", ""),
                    "completeness": result.get("step2_completeness", {}).get("reasoning", ""),
                    "consistency": result.get("step3_consistency", {}).get("reasoning", "")
                }
                
                # Calculate weighted final score
                final_score = (
                    criterion_scores["correctness"] * 0.5 +
                    criterion_scores["completeness"] * 0.3 +
                    criterion_scores["consistency"] * 0.2
                )
                
                return GEvalResult(
                    criterion_scores=criterion_scores,
                    criterion_reasoning=criterion_reasoning,
                    final_score=final_score,
                    overall_reasoning=result.get("overall_reasoning", ""),
                    evaluation_type="factual_accuracy",
                    confidence=result.get("confidence")
                )
                
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries:
                    self.logger.error(f"GEval factual accuracy evaluation failed: {e}")
                    return self._create_fallback_geval_result("factual_accuracy")
                continue
        
        return self._create_fallback_geval_result("factual_accuracy")
    
    async def evaluate_context_relevance_geval(
        self,
        question: str,
        contexts: List[str],
        llm_adapter: Any,
        max_retries: int = 2
    ) -> GEvalResult:
        """
        Evaluate context relevance using GEval multi-criteria approach.
        
        Criteria:
        1. Topic Alignment (40%): Does context address the question topic?
        2. Information Sufficiency (35%): Does context contain enough information?
        3. Answer Potential (25%): Can the question be answered using this context?
        """
        
        if not contexts:
            return self._create_fallback_geval_result("context_relevance")
        
        # Combine contexts with labels
        context_text = "\n\n".join([f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""You are an expert context relevance evaluator for RAG systems. Use systematic criteria to evaluate context quality.

**EVALUATION TASK**: Assess how relevant and useful the retrieved contexts are for answering the question.

**CHAIN-OF-THOUGHT EVALUATION**:

Let's evaluate this systematically:

**Step 1: Topic Alignment (Weight: 40%)**
- Task: Does the context address the right subject matter?
- Consider: Is this about the same topic? Are key entities/concepts mentioned?
- Scoring Rubric:
  * 90-100: Perfectly aligned with question topic, all key concepts present
  * 70-89: Good topic match, most relevant concepts present
  * 50-69: Partially related topic, some relevant concepts
  * 30-49: Tangentially related, few relevant concepts
  * 0-29: Completely off-topic or unrelated

**Step 2: Information Sufficiency (Weight: 35%)**
- Task: Does the context contain enough detail to answer?
- Consider: Is information comprehensive? Are details adequate? Is depth sufficient?
- Scoring Rubric:
  * 90-100: Rich, comprehensive information with excellent depth
  * 70-89: Good information coverage with adequate detail
  * 50-69: Basic information present but lacking depth
  * 30-49: Sparse information, many gaps
  * 0-29: Minimal or no useful information

**Step 3: Answer Potential (Weight: 25%)**
- Task: Can this context actually be used to construct an answer?
- Consider: Is information actionable? Can it directly support an answer?
- Scoring Rubric:
  * 90-100: Context directly enables complete answer construction
  * 70-89: Context supports most answer components
  * 50-69: Context partially supports answer construction
  * 30-49: Context provides minimal answer support
  * 0-29: Context cannot support answer construction

**INPUT DATA**:
Question: {question}

Contexts:
{context_text}

**OUTPUT FORMAT** (JSON only):
{{
  "step1_topic_alignment": {{
    "score": 0-100,
    "reasoning": "Detailed analysis of topic alignment"
  }},
  "step2_information_sufficiency": {{
    "score": 0-100,
    "reasoning": "Detailed analysis of information quality and depth"
  }},
  "step3_answer_potential": {{
    "score": 0-100,
    "reasoning": "Detailed analysis of answer construction potential"
  }},
  "final_score": "weighted average: (topic*0.4 + sufficiency*0.35 + potential*0.25)",
  "overall_reasoning": "Summary of context relevance assessment",
  "confidence": 0-100
}}

**YOUR EVALUATION**:"""
        
        for attempt in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                result = parse_json_response(
                    response.content,
                    self.logger,
                    "geval context relevance"
                )
                
                criterion_scores = {
                    "topic_alignment": float(result.get("step1_topic_alignment", {}).get("score", 0)),
                    "information_sufficiency": float(result.get("step2_information_sufficiency", {}).get("score", 0)),
                    "answer_potential": float(result.get("step3_answer_potential", {}).get("score", 0))
                }
                
                criterion_reasoning = {
                    "topic_alignment": result.get("step1_topic_alignment", {}).get("reasoning", ""),
                    "information_sufficiency": result.get("step2_information_sufficiency", {}).get("reasoning", ""),
                    "answer_potential": result.get("step3_answer_potential", {}).get("reasoning", "")
                }
                
                final_score = (
                    criterion_scores["topic_alignment"] * 0.4 +
                    criterion_scores["information_sufficiency"] * 0.35 +
                    criterion_scores["answer_potential"] * 0.25
                )
                
                return GEvalResult(
                    criterion_scores=criterion_scores,
                    criterion_reasoning=criterion_reasoning,
                    final_score=final_score,
                    overall_reasoning=result.get("overall_reasoning", ""),
                    evaluation_type="context_relevance",
                    confidence=result.get("confidence")
                )
                
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries:
                    self.logger.error(f"GEval context relevance evaluation failed: {e}")
                    return self._create_fallback_geval_result("context_relevance")
                continue
        
        return self._create_fallback_geval_result("context_relevance")
    
    async def evaluate_information_coverage_geval(
        self,
        question: str,
        ground_truth: str,
        answer: str,
        llm_adapter: Any,
        max_retries: int = 2
    ) -> GEvalResult:
        """
        Evaluate information coverage using GEval approach.
        
        Criteria:
        1. Key Facts Coverage (50%): Are essential facts present?
        2. Detail Coverage (30%): Is sufficient detail provided?
        3. Context Coverage (20%): Is relevant context/background included?
        """
        
        prompt = f"""You are an expert information coverage evaluator. Systematically assess how well the answer covers information from the reference.

**EVALUATION TASK**: Determine what percentage of essential information from the reference is covered in the answer.

**CHAIN-OF-THOUGHT EVALUATION**:

Let's analyze coverage step by step:

**Step 1: Key Facts Coverage (Weight: 50%)**
- Task: Are the most important facts from the reference present in the answer?
- Consider: Core facts, essential data points, critical information
- Scoring Rubric:
  * 90-100: All key facts present and accurate
  * 70-89: Most key facts present, minor ones missing
  * 50-69: Major key facts present but several missing
  * 30-49: Many key facts missing
  * 0-29: Most key facts absent

**Step 2: Detail Coverage (Weight: 30%)**
- Task: Is adequate detail and specificity provided?
- Consider: Numerical values, names, dates, descriptions, explanations
- Scoring Rubric:
  * 90-100: Rich detail matching reference depth
  * 70-89: Good detail, minor specifics missing
  * 50-69: Basic detail but lacking depth
  * 30-49: Minimal detail provided
  * 0-29: Almost no detail or specificity

**Step 3: Context Coverage (Weight: 20%)**
- Task: Is relevant background/context information included?
- Consider: Explanations, relationships, broader context, qualifying information
- Scoring Rubric:
  * 90-100: Comprehensive contextual information
  * 70-89: Good context, minor elements missing
  * 50-69: Basic context, lacking depth
  * 30-49: Minimal context provided
  * 0-29: Context largely absent

**INPUT DATA**:
Question: {question}
Reference (Ground Truth): {ground_truth}
Answer to Evaluate: {answer}

**OUTPUT FORMAT** (JSON only):
{{
  "step1_key_facts": {{
    "score": 0-100,
    "reasoning": "Analysis of key facts coverage with specific examples"
  }},
  "step2_detail_coverage": {{
    "score": 0-100,
    "reasoning": "Analysis of detail and specificity coverage"
  }},
  "step3_context_coverage": {{
    "score": 0-100,
    "reasoning": "Analysis of contextual information coverage"
  }},
  "final_score": "weighted average: (key_facts*0.5 + detail*0.3 + context*0.2)",
  "overall_reasoning": "Summary of information coverage assessment",
  "confidence": 0-100
}}

**YOUR EVALUATION**:"""
        
        for attempt in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(prompt)
                result = parse_json_response(
                    response.content,
                    self.logger,
                    "geval information coverage"
                )
                
                criterion_scores = {
                    "key_facts": float(result.get("step1_key_facts", {}).get("score", 0)),
                    "detail_coverage": float(result.get("step2_detail_coverage", {}).get("score", 0)),
                    "context_coverage": float(result.get("step3_context_coverage", {}).get("score", 0))
                }
                
                criterion_reasoning = {
                    "key_facts": result.get("step1_key_facts", {}).get("reasoning", ""),
                    "detail_coverage": result.get("step2_detail_coverage", {}).get("reasoning", ""),
                    "context_coverage": result.get("step3_context_coverage", {}).get("reasoning", "")
                }
                
                final_score = (
                    criterion_scores["key_facts"] * 0.5 +
                    criterion_scores["detail_coverage"] * 0.3 +
                    criterion_scores["context_coverage"] * 0.2
                )
                
                return GEvalResult(
                    criterion_scores=criterion_scores,
                    criterion_reasoning=criterion_reasoning,
                    final_score=final_score,
                    overall_reasoning=result.get("overall_reasoning", ""),
                    evaluation_type="information_coverage",
                    confidence=result.get("confidence")
                )
                
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries:
                    self.logger.error(f"GEval information coverage evaluation failed: {e}")
                    return self._create_fallback_geval_result("information_coverage")
                continue
        
        return self._create_fallback_geval_result("information_coverage")
    
    def _create_fallback_geval_result(self, evaluation_type: str) -> GEvalResult:
        """Create fallback GEval result when evaluation fails."""
        return GEvalResult(
            criterion_scores={},
            criterion_reasoning={},
            final_score=np.nan,
            overall_reasoning="Evaluation failed - unable to complete assessment",
            evaluation_type=evaluation_type,
            confidence=None
        )


class RagasProcessor:
    """
    Implements Ragas-style evaluation patterns for RAG systems:
    - Faithfulness: Factual consistency with retrieved context
    - Answer Relevance: How well answer addresses the question
    - Context Precision & Recall: Quality of retrieved contexts
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    async def evaluate_faithfulness_ragas(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        llm_adapter: Any,
        max_retries: int = 2
    ) -> RagasResult:
        """
        Evaluate faithfulness using Ragas approach:
        1. Extract statements from answer
        2. Verify each statement against contexts
        3. Calculate faithfulness score
        """
        
        if not contexts:
            return RagasResult(
                statement_scores=[],
                faithfulness_score=0.0,
                overall_reasoning="No contexts provided"
            )
        
        context_text = "\n\n".join([f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])
        
        # Step 1: Extract statements from answer
        extraction_prompt = f"""Extract all factual claims from the answer as independent, atomic statements.

**Requirements**:
- Each statement must be self-contained (no pronouns)
- Break down complex sentences into atomic claims
- Include ALL factual assertions

**Answer**: {answer}

**Output Format** (JSON array of strings):
["statement 1", "statement 2", ...]

**Extracted Statements**:"""
        
        statements = []
        for attempt in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(extraction_prompt)
                statements = parse_json_response(
                    response.content,
                    self.logger,
                    "ragas statement extraction"
                )
                if statements:
                    break
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries:
                    self.logger.error(f"Ragas statement extraction failed: {e}")
                    return RagasResult(
                        statement_scores=[],
                        faithfulness_score=np.nan,
                        overall_reasoning="Statement extraction failed"
                    )
                continue
        
        if not statements:
            return RagasResult(
                statement_scores=[],
                faithfulness_score=100.0,
                overall_reasoning="No statements to verify (empty answer)"
            )
        
        # Step 2: Verify each statement against contexts
        verification_prompt = f"""Verify if each statement can be inferred from the provided contexts.

**Verification Rules** (Ragas Faithfulness):
- Statement is SUPPORTED (1) if it can be directly inferred from context
- Statement is NOT SUPPORTED (0) if it cannot be verified or contradicts context
- Be strict: partial support counts as not supported

**Contexts**:
{context_text}

**Statements to Verify**:
{json.dumps(statements)}

**Output Format** (JSON):
{{
  "verifications": [
    {{
      "statement": "exact statement text",
      "supported": 1 or 0,
      "reasoning": "brief explanation",
      "supporting_context": "which context supports this (if supported)"
    }}
  ]
}}

**Your Verification**:"""
        
        statement_scores = []
        for attempt in range(max_retries + 1):
            try:
                response = await llm_adapter.ainvoke(verification_prompt)
                result = parse_json_response(
                    response.content,
                    self.logger,
                    "ragas verification"
                )
                statement_scores = result.get("verifications", [])
                if statement_scores:
                    break
            except (json.JSONDecodeError, Exception) as e:
                if attempt == max_retries:
                    self.logger.error(f"Ragas verification failed: {e}")
                    return RagasResult(
                        statement_scores=[],
                        faithfulness_score=np.nan,
                        overall_reasoning="Verification failed"
                    )
                continue
        
        # Step 3: Calculate faithfulness score
        if statement_scores:
            supported_count = sum(1 for s in statement_scores if s.get("supported") == 1)
            faithfulness_score = (supported_count / len(statement_scores)) * 100.0
        else:
            faithfulness_score = np.nan
        
        return RagasResult(
            statement_scores=statement_scores,
            faithfulness_score=faithfulness_score,
            overall_reasoning=f"Faithfulness: {supported_count}/{len(statement_scores)} statements supported by context"
        )
    
    async def evaluate_answer_support_ragas(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        llm_adapter: Any,
        max_retries: int = 2
    ) -> Tuple[float, str]:
        """
        Evaluate how well contexts support the answer using Ragas approach.
        Similar to faithfulness but focuses on answer support percentage.
        
        Returns: (support_percentage, reasoning)
        """
        
        result = await self.evaluate_faithfulness_ragas(
            question, answer, contexts, llm_adapter, max_retries
        )
        
        return result.faithfulness_score, result.overall_reasoning
