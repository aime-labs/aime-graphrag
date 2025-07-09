import re
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import datetime
from rouge_score import rouge_scorer
import asyncio

# --- Answer Correctness ---
class StatementsWithReason:
    def __init__(self, statement: str, reason: str):
        self.statement = statement
        self.reason = reason

class ClassificationWithReason:
    def __init__(self, TP=None, FP=None, FN=None):
        self.TP = TP if TP is not None else []
        self.FP = FP if FP is not None else []
        self.FN = FN if FN is not None else []

STATEMENT_GENERATOR_PROMPT = """
Generate concise independent statements from the given text that represent factual claims.
Respond ONLY with a JSON array of strings. Do not include any other text.

Example Input: 
"The sun is powered by nuclear fusion. This process creates light and heat."

Example Output:
["The sun is powered by nuclear fusion", "Nuclear fusion creates light and heat"]

Input Text:
{text}

Generated Statements:
"""

CORRECTNESS_PROMPT_TEMPLATE = """
Analyze statements from an answer compared to ground truth. Classify each as:
- TP (True Positive): Present in answer and supported by ground truth
- FP (False Positive): Present in answer but unsupported
- FN (False Negative): Missing from answer but present in ground truth

Provide JSON output with lists of TP, FP, FN objects containing 'statement' and 'reason'.

Examples:
{examples}

Current Analysis:
Question: "{question}"
Answer Statements: {answer}
Ground Truth Statements: {ground_truth}
"""

CORRECTNESS_EXAMPLES = [
    {
        "input": {
            "question": "What powers the sun and its primary function?",
            "answer": [
                "The sun is powered by nuclear fission",
                "Its primary function is providing light"
            ],
            "ground_truth": [
                "The sun is powered by nuclear fusion",
                "Fusion creates energy for heat and light",
                "Sunlight is essential for Earth's climate"
            ]
        },
        "output": {
            "TP": [{"statement": "Its primary function is providing light", "reason": "Matches ground truth about light"}],
            "FP": [{"statement": "The sun is powered by nuclear fission", "reason": "Contradicts fusion fact"}],
            "FN": [
                {"statement": "The sun is powered by nuclear fusion", "reason": "Missing correct power source"},
                {"statement": "Fusion creates energy for heat and light", "reason": "Missing energy creation detail"}
            ]
        }
    }
]

def fbeta_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)

async def generate_statements(llm, text: str, callbacks=None) -> List[str]:
    prompt = STATEMENT_GENERATOR_PROMPT.format(text=text)
    response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return []

async def calculate_factuality(llm, question: str, answer_stmts: List[str], gt_stmts: List[str], callbacks, beta: float) -> float:
    if not answer_stmts and not gt_stmts:
        return 1.0
    examples = "\n".join(
        f"Input: {json.dumps(ex['input'])}\nOutput: {json.dumps(ex['output'])}"
        for ex in CORRECTNESS_EXAMPLES
    )
    prompt = CORRECTNESS_PROMPT_TEMPLATE.format(
        examples=examples,
        question=question,
        answer=json.dumps(answer_stmts),
        ground_truth=json.dumps(gt_stmts)
    )
    response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
    try:
        classification = json.loads(response.content)
        tp = len(classification.get("TP", []))
        fp = len(classification.get("FP", []))
        fn = len(classification.get("FN", []))
        return fbeta_score(tp, fp, fn, beta)
    except (json.JSONDecodeError, TypeError):
        return 0.0

async def calculate_semantic_similarity(embeddings, answer: str, ground_truth: str) -> float:
    a_embed, gt_embed = await asyncio.gather(
        embeddings.aembed_query(answer),
        embeddings.aembed_query(ground_truth)
    )
    cosine_sim = np.dot(a_embed, gt_embed) / (
        np.linalg.norm(a_embed) * np.linalg.norm(gt_embed))
    return (cosine_sim + 1) / 2

async def compute_answer_correctness(
    question: str,
    answer: str,
    ground_truth: str,
    llm,
    embeddings,
    weights: List[float] = [0.75, 0.25],
    beta: float = 1.0,
    callbacks=None
) -> float:
    answer_statements = await generate_statements(llm, answer, callbacks)
    gt_statements = await generate_statements(llm, ground_truth, callbacks)
    factuality_score = await calculate_factuality(
        llm, question, answer_statements, gt_statements, callbacks, beta
    ) if weights[0] != 0 else 0.0
    similarity_score = await calculate_semantic_similarity(
        embeddings, answer, ground_truth
    ) if weights[1] != 0 else 0.0
    return float(np.average([factuality_score, similarity_score], weights=weights))

# --- Coverage ---
FACT_EXTRACTION_PROMPT = """
### Task
Extract distinct factual statements from the reference answer that could be independently verified.
Respond ONLY with a JSON object containing a "facts" list of strings.

### Example
Input:
  Question: "What causes seasons?"
  Reference: "Seasonal changes result from Earth's axial tilt. This tilt causes different hemispheres to receive varying sunlight."

Output:
{{
  "facts": [
    "Seasonal changes result from Earth's axial tilt",
    "The axial tilt causes different hemispheres to receive varying sunlight"
  ]
}}

### Actual Input
Question: "{question}"
Reference Answer: "{reference}"

### Your Response:
"""

FACT_COVERAGE_PROMPT = """
### Task
For each factual statement from the reference, determine if it's covered in the response.
Respond ONLY with a JSON object containing a "classifications" list. Each item should have:
- "statement": the exact fact from reference
- "attributed": 1 if covered, 0 if not

### Example
Response: "Seasons are caused by Earth's tilted axis"
Reference Facts: [
  "Seasonal changes result from Earth's axial tilt",
  "The axial tilt causes different hemispheres to receive varying sunlight"
]

Output:
{{
  "classifications": [
    {{"statement": "Seasonal changes result from Earth's axial tilt", "attributed": 1}},
    {{"statement": "The axial tilt causes different hemispheres to receive varying sunlight", "attributed": 0}}
  ]
}}

### Actual Input
Question: "{question}"
Response: "{response}"
Reference Facts: {facts}

### Your Response:
"""

def _validate_facts(facts: List) -> List[str]:
    return [str(f) for f in facts if f and str(f).strip()]

def _validate_classifications(classifications: List) -> List[Dict]:
    valid = []
    for item in classifications:
        try:
            if ("statement" in item and 
                "attributed" in item and item["attributed"] in {0, 1}):
                valid.append({
                    "statement": str(item["statement"]),
                    "attributed": int(item["attributed"])
                })
        except (TypeError, ValueError):
            continue
    return valid

async def _extract_facts(
    question: str,
    reference: str,
    llm,
    callbacks,
    max_retries: int
) -> List[str]:
    prompt = FACT_EXTRACTION_PROMPT.format(
        question=question,
        reference=reference[:3000]
    )
    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            data = json.loads(response.content)
            return _validate_facts(data.get("facts", []))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return []

async def _check_fact_coverage(
    question: str,
    facts: List[str],
    response: str,
    llm,
    callbacks,
    max_retries: int
) -> List[Dict]:
    prompt = FACT_COVERAGE_PROMPT.format(
        question=question,
        response=response[:3000],
        facts=json.dumps(facts)
    )
    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            data = json.loads(response.content)
            return _validate_classifications(data.get("classifications", []))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return []

async def compute_coverage_score(
    question: str,
    reference: str,
    response: str,
    llm,
    callbacks=None,
    max_retries: int = 2
) -> float:
    if not reference.strip():
        return 1.0
    facts = await _extract_facts(
        question, reference, llm, callbacks, max_retries
    )
    if not facts:
        return np.nan
    coverage = await _check_fact_coverage(
        question, facts, response, llm, callbacks, max_retries
    )
    if coverage:
        attributed = [c["attributed"] for c in coverage]
        return sum(attributed) / len(attributed)
    return np.nan

# --- Faithfulness ---
STATEMENT_GENERATION_PROMPT_FAITH = """
### Task
Break down the answer into atomic statements that are fully understandable without pronouns.
Respond ONLY with a JSON array of strings.

### Example
Question: "Who was Albert Einstein?"
Answer: "He was a German physicist known for relativity."
Output: ["Albert Einstein was a German physicist", "Albert Einstein is known for relativity"]

### Actual Input
Question: "{question}"
Answer: "{answer}"

### Generated Statements:
"""

FAITHFULNESS_EVALUATION_PROMPT = """
### Task
Judge if each statement can be directly inferred from the context. 
Respond ONLY with a JSON array of objects, each containing:
- "statement": the exact statement
- "verdict": 1 (supported) or 0 (not supported)
- "reason": brief explanation (1 sentence)

### Context
{context}

### Statements to Evaluate
{statements}

### Example Response
[
  {{"statement": "John is a computer science major", "verdict": 1, "reason": "Context says John studies Computer Science"}},
  {{"statement": "John works part-time", "verdict": 0, "reason": "No mention of employment in context"}}
]

### Your Response:
"""

def _validate_verdicts(verdicts: List) -> List[Dict]:
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

async def _generate_statements_faith(
    question: str,
    answer: str,
    llm,
    callbacks,
    max_retries: int
) -> List[str]:
    prompt = STATEMENT_GENERATION_PROMPT_FAITH.format(
        question=question[:500],
        answer=answer[:3000]
    )
    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            return json.loads(response.content)
        except json.JSONDecodeError:
            continue
    return []

async def _evaluate_statements(
    statements: List[str],
    context: str,
    llm,
    callbacks,
    max_retries: int
) -> List[Dict]:
    prompt = FAITHFULNESS_EVALUATION_PROMPT.format(
        context=context[:10000],
        statements=json.dumps(statements)[:5000]
    )
    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            return _validate_verdicts(json.loads(response.content))
        except (json.JSONDecodeError, TypeError):
            continue
    return []

async def compute_faithfulness_score(
    question: str,
    answer: str,
    contexts: List[str],
    llm,
    callbacks=None,
    max_retries: int = 2
) -> float:
    statements = await _generate_statements_faith(
        question, answer, llm, callbacks, max_retries
    )
    if not statements:
        return 1.0 if not answer.strip() else np.nan
    context_str = "\n".join(contexts)
    if not context_str.strip():
        return 0.0
    verdicts = await _evaluate_statements(
        statements, context_str, llm, callbacks, max_retries
    )
    if verdicts:
        supported = [v["verdict"] for v in verdicts]
        return sum(supported) / len(supported)
    return np.nan

# --- Context Relevance ---
CONTEXT_RELEVANCE_PROMPT = """
### Task
Evaluate the relevance of the Context for answering the Question using ONLY the information provided.
Respond ONLY with a number from 0-2. Do not explain.

### Rating Scale
0: Context has NO relevant information
1: Context has PARTIAL relevance
2: Context has RELEVANT information

### Question
{question}

### Context
{context}

### Rating:
"""

def _parse_rating(text: str) -> float:
    for token in text.split()[:8]:
        if token.isdigit() and 0 <= int(token) <= 2:
            return float(token)
    return None

async def _get_llm_rating(
    question: str,
    context: str,
    llm,
    callbacks,
    max_retries: int
) -> float:
    prompt = CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context)
    for _ in range(max_retries):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            return _parse_rating(response.content)
        except Exception:
            continue
    return None

async def compute_context_relevance(
    question: str,
    contexts: List[str],
    llm,
    callbacks=None,
    max_retries: int = 3
) -> float:
    if not question.strip() or not contexts or not any(c.strip() for c in contexts):
        return 0.0
    context_str = "\n".join(contexts)[:7000]
    if context_str.strip() == question.strip() or context_str.strip() in question:
        return 0.0
    rating1 = await _get_llm_rating(question, context_str, llm, callbacks, max_retries)
    rating2 = await _get_llm_rating(question, context_str, llm, callbacks, max_retries)
    scores = [r/2 for r in [rating1, rating2] if r is not None]
    if not scores:
        return np.nan
    return sum(scores) / len(scores)

# --- ROUGE ---
async def compute_rouge_score(
    answer: str,
    ground_truth: str,
    rouge_type: str = "rougeL",
    mode: str = "fmeasure"
) -> float:
    if not ground_truth.strip() or not answer.strip():
        return 0.0
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = scorer.score(ground_truth, answer)
    return getattr(scores[rouge_type], mode)

def exact_match(answer: str, ground_truth: str) -> float:
    """Return 1.0 if answer matches ground_truth exactly, else 0.0."""
    return float(answer.strip() == ground_truth.strip())

def f1_score(answer: str, ground_truth: str) -> float:
    """Compute F1 score between answer and ground_truth tokens."""
    answer_tokens = set(answer.lower().split())
    gt_tokens = set(ground_truth.lower().split())
    if not answer_tokens and not gt_tokens:
        return 1.0
    if not answer_tokens or not gt_tokens:
        return 0.0
    tp = len(answer_tokens & gt_tokens)
    fp = len(answer_tokens - gt_tokens)
    fn = len(gt_tokens - answer_tokens)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return 2 * precision * recall / (precision + recall + 1e-10)

def hallucination_rate(faithfulness: float) -> float:
    """Return 1.0 - faithfulness if faithfulness is not None, else nan."""
    return 1.0 - faithfulness if faithfulness is not None else float('nan')

async def extract_triples_llm(text: str, llm=None) -> list:
    """
    Extract triples (subject, predicate, object) from text using an LLM or a simple pattern.
    If llm is provided, use it to extract triples; otherwise, use a regex-based OpenIE-style extraction as a placeholder.
    Replace this with a real LLM-based extraction for production use.
    """
    if llm is not None:
        prompt = f"Extract all (subject, predicate, object) triples from the following text as a list of tuples.\nText: {text}"
        response = await llm.ainvoke(prompt)
        try:
            triples = eval(response.content) if isinstance(response.content, str) else response.content
            if isinstance(triples, list):
                return [tuple(t) for t in triples if isinstance(t, (list, tuple)) and len(t) == 3]
        except Exception:
            pass
        return []
    triple_pattern = re.compile(r'([A-Z][^\s]+)\s+([a-z]+)\s+([^\.]+)\.')
    triples = []
    for match in triple_pattern.finditer(text):
        subj, pred, obj = match.groups()
        triples.append((subj.strip(), pred.strip(), obj.strip()))
    return triples

async def compute_triple_metrics(answer: str, ground_truth: str, llm=None) -> dict:
    """
    Compute triple exact match and F1 between answer and ground_truth using extracted triples.
    """
    answer_triples = await extract_triples_llm(answer, llm)
    gt_triples = await extract_triples_llm(ground_truth, llm)
    set_a = set(answer_triples)
    set_g = set(gt_triples)
    if not set_g:
        return {"triple_em": float(not set_a), "triple_f1": 1.0 if not set_a else 0.0}
    tp = len(set_a & set_g)
    fp = len(set_a - set_g)
    fn = len(set_g - set_a)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    em = float(set_a == set_g)
    return {"triple_em": em, "triple_f1": f1}

async def compute_all_metrics(
    q: str,
    answer: str,
    gt: str,
    context_strs: List[str],
    llm_adapter: Any,
    embeddings_adapter: Any,
    logs: List[dict],
    method: str,
    log_error_fn = None
) -> Dict[str, Any]:
    """
    Compute all metrics for a given answer and log errors using the provided log_error_fn.
    """
    metrics = {}
    def log_error(message, q, method, level='error'):
        if log_error_fn:
            log_error_fn(logs, message, q, method, level)
    try:
        metrics['rouge_score'] = await compute_rouge_score(answer, gt)
    except Exception as e:
        metrics['rouge_score'] = None
        log_error(f'ROUGE metric failed: {str(e)}', q, method)
    try:
        metrics['answer_correctness'] = await compute_answer_correctness(q, answer, gt, llm_adapter, embeddings_adapter)
    except Exception as e:
        metrics['answer_correctness'] = None
        log_error(f'Answer correctness metric failed: {str(e)}', q, method)
    try:
        metrics['coverage_score'] = await compute_coverage_score(q, gt, answer, llm_adapter)
    except Exception as e:
        metrics['coverage_score'] = None
        log_error(f'Coverage score metric failed: {str(e)}', q, method)
    try:
        metrics['faithfulness'] = await compute_faithfulness_score(q, answer, context_strs, llm_adapter)
    except Exception as e:
        metrics['faithfulness'] = None
        log_error(f'Faithfulness metric failed: {str(e)}', q, method)
    try:
        metrics['context_relevance'] = await compute_context_relevance(q, context_strs, llm_adapter)
    except Exception as e:
        metrics['context_relevance'] = None
        log_error(f'Context relevance metric failed: {str(e)}', q, method)
    try:
        metrics['exact_match'] = exact_match(answer, gt)
    except Exception as e:
        metrics['exact_match'] = None
        log_error(f'Exact match metric failed: {str(e)}', q, method)
    try:
        metrics['f1_score'] = f1_score(answer, gt)
    except Exception as e:
        metrics['f1_score'] = None
        log_error(f'F1 score metric failed: {str(e)}', q, method)
    try:
        metrics['hallucination_rate'] = hallucination_rate(metrics.get('faithfulness'))
    except Exception as e:
        metrics['hallucination_rate'] = None
        log_error(f'Hallucination rate metric failed: {str(e)}', q, method)
    try:
        triple_metrics = await compute_triple_metrics(answer, gt, llm_adapter)
        metrics.update(triple_metrics)
    except Exception as e:
        metrics['triple_em'] = None
        metrics['triple_f1'] = None
        log_error(f'Triple metrics failed: {str(e)}. Triple metrics are not available.', q, method, level='warning')
    return metrics 