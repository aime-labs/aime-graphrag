# GraphRAG Evaluation Metrics Documentation

## Metrics Summary

| Category | Metric | Short Description |
|----------|--------|-------------------|
| **RAGAS** | `ragas_faithfulness` | Measures how factually consistent the answer is with retrieved context (detects hallucinations) |
| **RAGAS** | `ragas_context_precision` | Evaluates relevance of each retrieved context chunk to the question |
| **RAGAS** | `ragas_context_recall` | Measures how much required information from ground truth is present in contexts |
| **RAGAS** | `ragas_answer_relevance` | Measures how well the generated answer addresses the asked question |
| **RAGAS** | `ragas_score` | Weighted composite of all four RAGAS metrics for overall RAG quality |
| **Factual** | `factual_accuracy_percentage` | GEval-based correctness assessment comparing answer to ground truth |
| **Semantic** | `semantic_similarity_percentage` | Embedding-based cosine similarity between answer and ground truth |
| **Semantic** | `bert_score_f1` | Token-level contextual embedding similarity using BERT/DeBERTa |
| **Classification** | `correct_answers_count` | Count of answers classified as correct |
| **Classification** | `wrong_answers_count` | Count of answers classified as incorrect |
| **Classification** | `dont_know_answers_count` | Count of answers where model indicated uncertainty |

---

## Table of Contents

1. [RAGAS Faithfulness](#1-ragas-faithfulness-ragas_faithfulness)
2. [RAGAS Context Precision](#2-ragas-context-precision-ragas_context_precision)
3. [RAGAS Context Recall](#3-ragas-context-recall-ragas_context_recall)
4. [RAGAS Answer Relevance](#4-ragas-answer-relevance-ragas_answer_relevance)
5. [RAGAS Composite Score](#5-ragas-composite-score-ragas_score)
6. [Factual Accuracy Percentage](#6-factual-accuracy-percentage-factual_accuracy_percentage)
7. [Semantic Similarity Percentage](#7-semantic-similarity-percentage-semantic_similarity_percentage)
8. [BERTScore F1](#8-bertscore-f1-bert_score_f1)
9. [Correct Answers Count](#9-correct-answers-count-correct_answers_count)
10. [Wrong Answers Count](#10-wrong-answers-count-wrong_answers_count)
11. [Don't Know Answers Count](#11-dont-know-answers-count-dont_know_answers_count)

---

## 1. RAGAS Faithfulness (`ragas_faithfulness`)

### What is it?
Faithfulness measures how factually consistent the generated answer is with the retrieved context. It evaluates whether all claims made in the answer can be traced back to and supported by the provided context documents.

### Relevance to Evaluation
In RAG systems, faithfulness is critical because:
- It detects **hallucinations** - when the model generates information not present in the context
- It ensures the answer is **grounded** in the retrieved evidence
- It measures the system's ability to stay truthful to source material
- A high faithfulness score indicates the RAG system is not fabricating information

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| `answer` | `str` | The generated answer to evaluate |
| `contexts` | `List[str]` | Retrieved context documents |
| `question` | `str` | Original question (used for context) |

### Calculation

**Formula:**
$$\text{Faithfulness} = \frac{\text{Number of supported claims}}{\text{Total number of claims}} \times 100$$

**Step-by-Step Process:**
1. **Claim Extraction**: Extract all atomic factual claims from the answer
2. **Claim Verification**: For each claim, verify if it's supported by the context
3. **Score Computation**: Calculate the ratio of supported claims to total claims

### Pseudo Code
```python
def compute_faithfulness(answer: str, contexts: List[str]) -> float:
    # Step 1: Extract claims from answer
    claims = llm_extract_claims(answer)
    
    if len(claims) == 0:
        return 100.0  # Empty answer = no unsupported claims
    
    # Step 2: Verify each claim against context
    supported_count = 0
    for claim in claims:
        is_supported = llm_verify_claim(claim, contexts)
        if is_supported:
            supported_count += 1
    
    # Step 3: Calculate score
    faithfulness_score = (supported_count / len(claims)) * 100.0
    return faithfulness_score
```

### Prompts Used

**Claim Extraction Prompt:**
```
You are an expert at extracting factual claims from text.
Your task is to break down the answer into atomic, self-contained factual statements.

**INSTRUCTIONS:**
1. Extract ALL factual claims from the answer
2. Each claim must be:
   - ATOMIC: Contains exactly one piece of information
   - SELF-CONTAINED: Makes sense without the original question
   - VERIFIABLE: Can be checked against external knowledge
3. Replace pronouns with their referents
4. Do NOT include opinions, questions, or hypotheticals
5. Do NOT modify or paraphrase - preserve the original meaning

**INPUT:**
Question: In the narrative of 'An Unsentimental Journey through Cornwall', which plant 
known scientifically as Erica vagans is also referred to by another common name, and what is that name?
Answer: Cornish heath

**OUTPUT:**
Respond with ONLY a JSON array of strings. Each string is one atomic claim.
```

**Claim Verification Prompt:**
```
Task: Analyze the given claim and context carefully. Determine if the claim is supported by the context.

Instructions:
1. Consider only explicit information in the context
2. A claim is supported if it can be directly inferred from the context
3. A claim is NOT supported if the context contradicts it OR doesn't mention it

Context: {combined_contexts}

Claims to verify: {claims}

For each claim, output JSON with:
- claim: The claim text
- verdict: 1 if supported, 0 if not supported
- reason: Brief explanation
```

### Output
- **Type**: `float`
- **Range**: `0-100`
- **Interpretation**: Higher is better (100 = perfectly faithful to context)

---

## 2. RAGAS Context Precision (`ragas_context_precision`)

### What is it?
Context Precision measures the relevance of each retrieved context chunk to answering the question. It evaluates whether the contexts retrieved by the RAG system are actually useful for generating a good answer.

### Relevance to Evaluation
Context Precision is important because:
- It evaluates the **retrieval component** of the RAG pipeline
- High precision means retrieved contexts are relevant, not noisy
- It identifies if the retriever is fetching irrelevant documents
- Poor context precision wastes LLM context window and can confuse the generation

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| `question` | `str` | The question being asked |
| `contexts` | `List[str]` | Retrieved context documents (in order) |
| `ground_truth` | `str` | Reference answer (optional, for enhanced evaluation) |

### Calculation

**Formula:**
$$\text{Context Precision} = \frac{1}{N} \sum_{i=1}^{N} \text{relevance}_i \times 100$$

Where $N$ is the number of context chunks and $\text{relevance}_i \in \{0, 1\}$.

**Alternative Precision@K (when order matters):**
$$\text{Precision@K} = \frac{\sum_{k=1}^{K} (\text{relevance}_k \times \text{Precision@k})}{|\text{relevant contexts in top K}|}$$

### Pseudo Code
```python
def compute_context_precision(question: str, contexts: List[str]) -> float:
    if not contexts:
        return 0.0
    
    # Evaluate relevance of each context chunk
    relevance_verdicts = []
    for context in contexts:
        verdict = llm_evaluate_relevance(question, context)
        relevance_verdicts.append(verdict)  # 0 or 1
    
    # Simple average (all contexts weighted equally)
    precision_score = (sum(relevance_verdicts) / len(relevance_verdicts)) * 100.0
    return precision_score
```

### Output
- **Type**: `float`
- **Range**: `0-100`
- **Interpretation**: Higher is better (100 = all contexts are relevant)

### Real Example from Llama3_test

**Question ID**: `Novel-73586ddc`
**Question**: *"In the narrative of 'An Unsentimental Journey through Cornwall', which plant known scientifically as Erica vagans is also referred to by another common name?"*

**Ground Truth**: `Cornish heath`

**Metric Results by Method**:
| Method | Context Precision Score |
|--------|------------------------|
| local_search | NaN (not calculable) |
| basic_search | 0.0 |
| llm_with_context | N/A |

**LLM Prompt Used (Context Relevance Evaluation)**:
```
You are evaluating retrieved contexts for a RAG system.
For each context, determine if it contains information useful for answering the question.

**QUESTION:** In the narrative of 'An Unsentimental Journey through Cornwall', which plant 
known scientifically as Erica vagans is also referred to by another common name?

**GROUND TRUTH ANSWER:** Cornish heath

**CONTEXTS TO EVALUATE:**
**CONTEXT 1:** [Source 901]: ...THE END. LONDON: R. CLAY, SONS, AND TAYLOR...
**CONTEXT 2:** [Source 802]: ...its vegetation includes nothing bigger than the _erica vagans_
--the lovely Cornish heath, lilac, flesh- and white which will grow nowhere else...
[...16 contexts total...]

**EVALUATION CRITERIA:**
A context is RELEVANT if it:
- Contains information that directly helps answer the question
- Provides facts, definitions, or explanations related to the query
- Would be useful as supporting evidence for an answer

**OUTPUT FORMAT:**
JSON array with exactly 16 elements, each containing:
{
  "context_index": 1-16,
  "is_relevant": true or false,
  "relevance_score": 0-100,
  "reasoning": "brief explanation"
}
```

**Raw LLM Response (partial)**:
```json
[
  {"context_index": 1, "is_relevant": false, "relevance_score": 5, 
   "reasoning": "Only contains the ebook ending; no mention of Erica vagans or common name."},
  {"context_index": 2, "is_relevant": true, "relevance_score": 98, 
   "reasoning": "Explicitly states that Erica vagans is 'the lovely Cornish heath', providing both scientific and common names."},
  {"context_index": 7, "is_relevant": true, "relevance_score": 97, 
   "reasoning": "Repeats the passage linking Erica vagans to 'the lovely Cornish heath'"}
]
```

**Interpretation**: Context 2 and 7 were highly relevant (scores 97-98) as they explicitly mentioned the plant. Most other contexts scored <10 as they were irrelevant book metadata or unrelated passages.

---

## 3. RAGAS Context Recall (`ragas_context_recall`)

### What is it?
Context Recall measures how much of the required information (from ground truth) is actually present in the retrieved contexts. It evaluates the completeness of the retrieval - are all necessary facts retrievable from the context?

### Relevance to Evaluation
Context Recall matters because:
- It measures if the retriever **found all necessary information**
- Low recall means the answer cannot be complete regardless of generation quality
- It identifies gaps in the knowledge retrieval process
- Critical for answers that require multiple pieces of information

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| `ground_truth` | `str` | Reference answer containing required information |
| `contexts` | `List[str]` | Retrieved context documents |
| `question` | `str` | The question (for context) |

### Calculation

**Formula:**
$$\text{Context Recall} = \frac{\text{GT statements attributable to context}}{\text{Total GT statements}} \times 100$$

**Process:**
1. Decompose ground truth into atomic statements
2. For each statement, check if it can be attributed to retrieved contexts
3. Calculate the percentage of attributable statements

### Pseudo Code
```python
def compute_context_recall(ground_truth: str, contexts: List[str]) -> float:
    # Step 1: Extract statements from ground truth
    gt_statements = llm_extract_statements(ground_truth)
    
    if not gt_statements:
        return 100.0  # No statements to recall
    
    if not contexts:
        return 0.0  # No context means no recall
    
    # Step 2: Check attribution for each statement
    attributable_count = 0
    for statement in gt_statements:
        can_attribute = llm_check_attribution(statement, contexts)
        if can_attribute:
            attributable_count += 1
    
    # Step 3: Calculate recall
    recall_score = (attributable_count / len(gt_statements)) * 100.0
    return recall_score
```

### Output
- **Type**: `float`
- **Range**: `0-100`
- **Interpretation**: Higher is better (100 = all required info is in context)

### Real Example from Llama3_test

**Question ID**: `Novel-73586ddc`  
**Question**: *"In the narrative of 'An Unsentimental Journey through Cornwall', which plant known scientifically as Erica vagans is also referred to by another common name?"*

**Ground Truth**: `Cornish heath`

**Metric Results by Method**:
| Method | Context Recall Score |
|--------|---------------------|
| local_search | 100.0 |
| basic_search | 0.0 |
| llm_with_context | N/A |

**LLM Prompt Used (Ground Truth Statement Extraction)**:
```
You are extracting key information from a reference answer.
Break down the answer into individual statements that convey important information.

**INSTRUCTIONS:**
1. Extract ALL distinct pieces of information from the answer
2. Each statement should be atomic and self-contained
3. Include facts, relationships, quantities, names, dates, etc.
4. Preserve the original meaning without adding interpretations
5. Make statements complete enough to be verifiable independently

**QUESTION:** In the narrative of 'An Unsentimental Journey through Cornwall', which plant 
known scientifically as Erica vagans is also referred to by another common name?

**GROUND TRUTH ANSWER:** Cornish heath

**OUTPUT:**
Respond with ONLY a JSON array of strings, where each string is one key statement.
```
**Attribution Verification Prompt:**
```
Given a statement from the ground truth and a set of contexts, determine if the statement can be attributed to (found in or inferred from) the contexts.

Statement: {statement}

Contexts:
{numbered_contexts}

For each statement, output JSON:
{
  "statement": "the statement",
  "attributed": 1 if found in context else 0,
  "supporting_context": "Context X" or null
}
```

**Raw LLM Response**:
```json
["Cornish heath is the common name for Erica vagans."]
```

**Subsequent Attribution Check**:
The system then checks if this statement can be found in the retrieved contexts. For `local_search`, the context contained: *"the _erica vagans_--the lovely Cornish heath"* - directly attributing the statement, yielding 100% recall.

**Interpretation**: 
- 100% context recall means all ground truth information was retrievable from context
- 0% recall indicates the necessary information was not in the retrieved documents

---

## 4. RAGAS Answer Relevance (`ragas_answer_relevance`)

### What is it?
Answer Relevance measures how well the generated answer actually addresses the asked question. It evaluates whether the answer is on-topic and provides information that the question is seeking.

### Relevance to Evaluation
Answer Relevance is crucial because:
- Detects **off-topic or tangential answers**
- Identifies when answers are technically correct but don't address the question
- Measures alignment between question intent and answer content
- Catches verbose answers that bury relevant information in irrelevant content

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| `question` | `str` | The original question |
| `answer` | `str` | The generated answer |

### Calculation

**Method: Question Generation & Similarity**

The RAGAS approach generates synthetic questions from the answer and measures similarity:

$$\text{Answer Relevance} = \text{mean}\left(\text{cosine\_sim}(q_{original}, q_{generated})\right) \times 100$$

**Alternative Direct Assessment:**
$$\text{Answer Relevance} = \text{LLM\_score}(question, answer) \times 100$$

### Pseudo Code
```python
def compute_answer_relevance(question: str, answer: str) -> float:
    # Method 1: Question Generation Approach
    # Generate N questions that the answer would address
    generated_questions = llm_generate_questions(answer, n=3)
    
    if not generated_questions:
        # Fallback to direct assessment
        return llm_direct_relevance_score(question, answer)
    
    # Get embeddings
    original_embedding = get_embedding(question)
    
    similarities = []
    for gen_q in generated_questions:
        gen_embedding = get_embedding(gen_q)
        sim = cosine_similarity(original_embedding, gen_embedding)
        similarities.append(sim)
    
    # Average similarity
    relevance_score = (sum(similarities) / len(similarities)) * 100.0
    return relevance_score
```

### Prompts Used

**Question Generation Prompt:**
```
Generate 3 questions that the following answer would be a good response to.

Answer: {answer}

Requirements:
- Questions should be natural and coherent
- Each question should be answerable by the given answer
- Questions should cover different aspects of the answer

Output JSON array: ["question1", "question2", "question3"]
```

**Direct Relevance Assessment Prompt:**
```
Evaluate how well the answer addresses the question.

Question: {question}
Answer: {answer}

Scoring Rubric:
- 1.0: Answer directly and completely addresses the question
- 0.75: Answer mostly addresses the question with minor gaps
- 0.5: Answer partially addresses the question
- 0.25: Answer tangentially related to the question
- 0.0: Answer is completely irrelevant to the question

Output JSON:
{
  "score": 0.0-1.0,
  "reasoning": "Brief explanation"
}
```

### Output
- **Type**: `float`
- **Range**: `0-100`
- **Interpretation**: Higher is better (100 = perfectly relevant answer)

### Real Example from Llama3_test

**Question ID**: `Novel-73586ddc`  
**Question**: *"In the narrative of 'An Unsentimental Journey through Cornwall', which plant known scientifically as Erica vagans is also referred to by another common name, and what is that name?"*

**Answer**: `Cornish heath`

**Metric Results by Method**:
| Method | Answer Relevance Score |
|--------|----------------------|
| local_search | 83.27 |
| basic_search | 83.27 |
| llm_with_context | 82.29 |

**LLM Prompt Used (Question Generation)**:
```
You are generating questions that could be answered by the given text.
Think about what questions someone would need to ask to receive this answer.

**INSTRUCTIONS:**
1. Generate 3 different questions that this answer addresses
2. Questions should be diverse and cover different aspects
3. Questions should be natural and well-formed
4. The answer should be a valid response to each question

**ANSWER:**
Cornish heath

**OUTPUT:**
Respond with ONLY a JSON array of exactly 3 question strings.
```

**Raw LLM Response**:
```json
[
    "What is the name of the plant that is the county flower of Cornwall?",
    "Which species of heath is native to Cornwall and known as Erica vagans?",
    "What plant is commonly referred to as Cornish heath?"
]
```

**Similarity Calculation**:
- Original question embedding compared with each generated question embedding
- Cosine similarity averaged across 3 generated questions
- Result: 82-83% relevance (good alignment between answer and question intent)

**Interpretation**: The score of ~83% indicates the answer is highly relevant to the question, though not perfectly aligned (100%). The generated questions show the answer could address various botanical queries about Cornish heath.

---

## 5. RAGAS Composite Score (`ragas_score`)

### What is it?
The RAGAS Composite Score is a weighted combination of all four RAGAS metrics, providing a single holistic score for RAG system quality.

### Relevance to Evaluation
The composite score is useful because:
- Provides a **single metric** for overall RAG quality
- Balances retrieval quality (precision/recall) with generation quality (faithfulness/relevance)
- Enables quick comparison between different RAG configurations
- Weights can be adjusted based on application priorities

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| `faithfulness` | `float` | Faithfulness score (0-100) |
| `context_precision` | `float` | Context Precision score (0-100) |
| `context_recall` | `float` | Context Recall score (0-100) |
| `answer_relevance` | `float` | Answer Relevance score (0-100) |

### Calculation

**Formula:**
$$\text{RAGAS Score} = w_f \cdot F + w_{cp} \cdot CP + w_{cr} \cdot CR + w_{ar} \cdot AR$$

**Default Weights:**
| Metric | Weight | Rationale |
|--------|--------|-----------|
| Faithfulness ($w_f$) | 0.30 | Hallucination prevention is critical |
| Context Precision ($w_{cp}$) | 0.20 | Retrieval efficiency |
| Context Recall ($w_{cr}$) | 0.20 | Information completeness |
| Answer Relevance ($w_{ar}$) | 0.30 | Answer quality |

**Total**: $0.30 + 0.20 + 0.20 + 0.30 = 1.00$

### Pseudo Code
```python
def compute_ragas_score(
    faithfulness: float,
    context_precision: float, 
    context_recall: float,
    answer_relevance: float,
    weights: Dict[str, float] = None
) -> float:
    
    if weights is None:
        weights = {
            'faithfulness': 0.30,
            'context_precision': 0.20,
            'context_recall': 0.20,
            'answer_relevance': 0.30
        }
    
    scores = {
        'faithfulness': faithfulness,
        'context_precision': context_precision,
        'context_recall': context_recall,
        'answer_relevance': answer_relevance
    }
    
    # Handle NaN values - exclude from calculation
    valid_scores = {k: v for k, v in scores.items() if not math.isnan(v)}
    
    if not valid_scores:
        return float('nan')
    
    # Renormalize weights for valid scores only
    total_weight = sum(weights[k] for k in valid_scores.keys())
    
    weighted_sum = sum(
        scores[k] * (weights[k] / total_weight) 
        for k in valid_scores.keys()
    )
    
    return weighted_sum
```

### Output
- **Type**: `float`
- **Range**: `0-100`
- **Interpretation**: Higher is better (100 = perfect RAG system)

### Real Example from Llama3_test

**Question ID**: `Novel-73586ddc` - Erica vagans question

**Composite Score by Method**:
| Method | Faithfulness | Context Precision | Context Recall | Answer Relevance | **RAGAS Score** |
|--------|-------------|-------------------|----------------|------------------|-----------------|
| local_search | 100.0 | NaN | 100.0 | 83.27 | **93.73** |
| basic_search | 0.0 | 0.0 | 0.0 | 83.27 | **24.98** |
| llm_with_context | N/A | N/A | N/A | 82.29 | **82.29** |

**Score Calculation for local_search**:
```
RAGAS Score = (0.30 × 100.0) + (0.20 × NaN excluded) + (0.20 × 100.0) + (0.30 × 83.27)
            = 30.0 + 20.0 + 24.98 (renormalized)
            = 93.73
```

**Aggregate Statistics Across All 8 Questions**:
| Method | Avg RAGAS Score | Best Score | Worst Score |
|--------|----------------|------------|-------------|
| local_search | 89.11 | 95.33 | 71.33 |
| llm_with_context | 85.57 | 96.69 | 71.79 |
| basic_search | 43.68 | 75.51 | 20.41 |

**Interpretation**: The composite score shows `local_search` consistently outperforms other methods, with the GraphRAG knowledge graph providing better retrieval than basic search approaches.

---

## 6. Factual Accuracy Percentage (`factual_accuracy_percentage`)

### What is it?
Factual Accuracy measures the correctness of the generated answer compared to the ground truth reference answer using a multi-criteria GEval (LLM-as-a-Judge) approach with Chain-of-Thought reasoning.

### Relevance to Evaluation
Factual accuracy is essential because:
- Directly measures **answer correctness**
- Uses structured evaluation criteria for consistency
- Provides interpretable reasoning for the score
- Goes beyond simple text matching to semantic correctness

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| `question` | `str` | The question being answered |
| `answer` | `str` | The generated answer |
| `ground_truth` | `str` | The reference correct answer |

### Calculation

**GEval Multi-Criteria Approach:**

The metric uses three weighted criteria:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Correctness** | 50% | Are the facts accurate? |
| **Completeness** | 30% | Is all important information included? |
| **Consistency** | 20% | Is the answer internally consistent? |

**Formula:**
$$\text{Factual Accuracy} = 0.5 \times C_{correct} + 0.3 \times C_{complete} + 0.2 \times C_{consist}$$

### Pseudo Code
```python
def compute_factual_accuracy_percentage(
    question: str,
    answer: str,
    ground_truth: str
) -> str:
    
    # Use GEval processor for multi-criteria evaluation
    geval_result = await geval_processor.evaluate_factual_accuracy_geval(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        llm_adapter=llm
    )
    
    # Get weighted final score
    final_score = geval_result.final_score
    
    # Convert to letter grade classification
    if final_score >= 80:
        return "A"  # Excellent
    elif final_score >= 60:
        return "B"  # Good
    elif final_score >= 40:
        return "C"  # Moderate
    elif final_score >= 20:
        return "D"  # Poor
    else:
        return "E"  # Very Poor
```

### Prompt Used

**GEval Factual Accuracy Prompt:**
```
You are an expert factual accuracy evaluator. Use a systematic, step-by-step approach to evaluate the answer against the ground truth.

**EVALUATION TASK**: Assess factual accuracy of the candidate answer compared to the reference answer.

**CHAIN-OF-THOUGHT EVALUATION**:

**Step 1: Correctness Analysis (Weight: 50%)**
- Task: Evaluate if all factual claims in the answer are accurate
- Scoring Rubric:
  * 90-100: All facts are completely accurate, no errors
  * 70-89: Minor inaccuracies that don't change core meaning
  * 50-69: Some notable factual errors but main points correct
  * 30-49: Multiple significant factual errors
  * 0-29: Predominantly incorrect or fabricated facts

**Step 2: Completeness Analysis (Weight: 30%)**
- Task: Evaluate if all essential information from ground truth is present
- Scoring Rubric:
  * 90-100: All essential information present with appropriate detail
  * 70-89: Most key information present, minor details missing
  * 50-69: Core information present but lacking important details
  * 30-49: Significant gaps in essential information
  * 0-29: Most essential information missing

**Step 3: Consistency Analysis (Weight: 20%)**
- Task: Evaluate internal logical consistency of the answer
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

**OUTPUT FORMAT** (JSON):
{
  "step1_correctness": {"score": 0-100, "reasoning": "..."},
  "step2_completeness": {"score": 0-100, "reasoning": "..."},
  "step3_consistency": {"score": 0-100, "reasoning": "..."},
  "final_score": weighted_average,
  "overall_reasoning": "Summary",
  "confidence": 0-100
}
```

### Output
- **Type**: `str`
- **Values**: `"A"`, `"B"`, `"C"`, `"D"`, `"E"`
- **Interpretation**:
  - `A` (≥80%): Excellent accuracy
  - `B` (60-79%): Good accuracy
  - `C` (40-59%): Moderate accuracy
  - `D` (20-39%): Poor accuracy
  - `E` (<20%): Very poor accuracy

### Real Example from Llama3_test

**Question ID**: `Novel-73586ddc` - Erica vagans question

**Question**: *"In the narrative of 'An Unsentimental Journey through Cornwall', which plant known scientifically as Erica vagans is also referred to by another common name, and what is that name?"*

**Ground Truth**: `Cornish heath`  
**Candidate Answer**: `Cornish heath`

**Metric Results by Method**:
| Method | Factual Accuracy Grade |
|--------|----------------------|
| local_search | A |
| basic_search | A |
| llm_with_context | A |


**Contrasting Example - Wrong Answer**:

**Question ID**: `Novel-74440a6a`
**Question**: *"Within the account of the royal visit to St. Michael's Mount in Cornwall, who is identified as the person who married Princess Frederica of Hanover?"*

| Method | Ground Truth | Answer | Grade |
|--------|--------------|--------|-------|
| llm_with_context | Baron Alphonse | (incorrect answer) | **E** |
| local_search | Baron Alphonse | Baron Alphonse | **A** |

**Distribution Across All 24 Results**:
| Grade | Count | Percentage |
|-------|-------|------------|
| A | 12 | 50% |
| B | 1 | 4% |
| C | 3 | 12.5% |
| D | 4 | 17% |
| E | 4 | 17% |

---

## 7. Semantic Similarity Percentage (`semantic_similarity_percentage`)

### What is it?
Semantic Similarity measures how semantically close the generated answer is to the ground truth reference, using embedding-based cosine similarity. Unlike text matching, this captures meaning equivalence even with different wording.

### Relevance to Evaluation
Semantic similarity is valuable because:
- Captures **meaning** rather than exact text match
- Allows for paraphrasing and different phrasing
- Fast computation using pre-computed embeddings
- Good baseline metric for answer quality

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| `answer` | `str` | The generated answer |
| `ground_truth` | `str` | The reference correct answer |

### Calculation

**Formula:**
$$\text{Semantic Similarity} = \frac{\vec{a} \cdot \vec{g}}{||\vec{a}|| \times ||\vec{g}||} \times 100$$

Where:
- $\vec{a}$ = embedding vector of the answer
- $\vec{g}$ = embedding vector of the ground truth

**Range Normalization:**
Cosine similarity naturally ranges from -1 to 1, but text embeddings typically produce positive values. The result is normalized to 0-100%.

### Pseudo Code
```python
def compute_semantic_similarity_percentage(
    answer: str,
    ground_truth: str,
    embedding_model: Any
) -> float:
    
    # Handle empty inputs
    if not answer or not ground_truth:
        return 0.0
    
    if not answer.strip() or not ground_truth.strip():
        return 0.0
    
    # Get embeddings
    answer_embedding = embedding_model.embed(answer)
    ground_truth_embedding = embedding_model.embed(ground_truth)
    
    # Convert to numpy arrays
    answer_vec = np.array(answer_embedding)
    ground_truth_vec = np.array(ground_truth_embedding)
    
    # Calculate cosine similarity
    dot_product = np.dot(answer_vec, ground_truth_vec)
    norm_a = np.linalg.norm(answer_vec)
    norm_g = np.linalg.norm(ground_truth_vec)
    
    if norm_a == 0 or norm_g == 0:
        return 0.0
    
    cosine_sim = dot_product / (norm_a * norm_g)
    
    # Convert to percentage (0-100 scale)
    # Cosine similarity for text is typically 0-1, rarely negative
    similarity_percentage = max(0, cosine_sim) * 100.0
    
    return round(similarity_percentage, 2)
```

### Embedding Model
The pipeline uses the configured embedding model (e.g., `text-embedding-ada-002`, `all-MiniLM-L6-v2`, or custom embedding endpoint).

### Output
- **Type**: `float`
- **Range**: `0-100`
- **Interpretation**: Higher is better (100 = identical semantic meaning)

### Real Example from Llama3_test

**Question ID**: `Novel-73586ddc` - Erica vagans question

**Ground Truth**: `Cornish heath`  
**Candidate Answer**: `Cornish heath`

**Metric Results by Method**:
| Method | Semantic Similarity % |
|--------|----------------------|
| local_search | 100.0 |
| basic_search | 100.0 |
| llm_with_context | 96.72 |

**Calculation**:
```python
# Exact match case
answer_embedding = embed("Cornish heath")      # → [0.23, -0.15, 0.89, ...]
ground_truth_embedding = embed("Cornish heath") # → [0.23, -0.15, 0.89, ...]

cosine_sim = dot_product / (norm_a × norm_g) = 1.0
semantic_similarity = 1.0 × 100 = 100.0%
```

**Varied Examples Across Questions**:

| Question | Ground Truth | Answer | Similarity |
|----------|--------------|--------|------------|
| Erica vagans | "Cornish heath" | "Cornish heath" | 100.0% |
| Royal visit | "Baron Alphonse" | "Baron Alphonse" | 99.99% |
| King Arthur | (long passage) | (similar answer) | 87.97% |
| Fish diary | (creative answer) | (different creative) | 90.75% |

**Interpretation**: 
- 100% indicates identical or near-identical text
- 90-99% indicates semantically equivalent with minor variations
- 80-89% indicates core meaning preserved with different phrasing
- <80% indicates significant semantic divergence

---

## 8. BERTScore F1 (`bert_score_f1`)

### What is it?
BERTScore computes the similarity of two sentences using contextual embeddings from BERT models. Unlike simple embedding similarity, BERTScore considers token-level alignments to provide precision, recall, and F1 scores for semantic similarity.

### Relevance to Evaluation
BERTScore is valuable because:
- Uses **contextual embeddings** rather than static word vectors
- Provides **token-level matching** for more nuanced comparison
- Correlates well with human judgment for text similarity
- Captures paraphrase similarity better than exact matching

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| `answer` | `str` | The generated answer (candidate) |
| `ground_truth` | `str` | The reference correct answer |

### Calculation

**Formula:**
$$\text{BERTScore}_{F1} = 2 \times \frac{P \times R}{P + R}$$

Where:
- $P$ (Precision) = How many tokens in candidate match tokens in reference
- $R$ (Recall) = How many tokens in reference are covered by candidate

**Token Matching**: Uses cosine similarity of BERT embeddings with greedy maximum matching.

### Pseudo Code
```python
def compute_bert_score_f1(answer: str, ground_truth: str) -> float:
    from bert_score import score
    
    # Handle edge cases
    if not answer or not ground_truth:
        return 0.0
    
    # Compute BERTScore using microsoft/deberta-xlarge-mnli
    P, R, F1 = score(
        cands=[answer],
        refs=[ground_truth],
        model_type='microsoft/deberta-xlarge-mnli',
        verbose=False
    )
    
    # Convert to percentage (0-100 scale)
    bert_f1_percentage = F1.item() * 100.0
    return bert_f1_percentage
```

### Model Used
- **Default Model**: `microsoft/deberta-xlarge-mnli`
- This model provides high-quality contextual embeddings for semantic similarity

### Output
- **Type**: `float`
- **Range**: `0-100`
- **Interpretation**: Higher is better (100 = perfect token-level alignment)

### Real Example from Llama3_test

**Question ID**: `Novel-73586ddc` - Erica vagans question

**Ground Truth**: `Cornish heath`  
**Candidate Answer**: `Cornish heath`

**Metric Results by Method**:
| Method | BERTScore F1 |
|--------|-------------|
| local_search | 99.99% |
| basic_search | 99.99% |
| llm_with_context | 72.38% |

**Varied Examples Across Questions**:

| Question Type | Ground Truth (abbrev) | Answer (abbrev) | BERTScore F1 |
|---------------|----------------------|-----------------|--------------|
| Fact Retrieval (exact match) | "Cornish heath" | "Cornish heath" | 99.99% |
| Fact Retrieval (exact match) | "Baron Alphonse" | "Baron Alphonse" | 100.0% |
| Complex Reasoning | (long passage) | (similar long answer) | 16.23% |
| Creative Generation | (diary entry) | (different diary) | 6.62% |

**Interpretation**:
- **99-100%**: Near-identical text with perfect token alignment
- **70-99%**: Very similar content with minor word choice differences
- **30-70%**: Similar meaning but different phrasing/structure
- **0-30%**: Very different text (common for creative/long-form answers where multiple valid responses exist)

**Note**: BERTScore tends to be lower for creative generation tasks where many valid answers exist, even when the answer is correct. It's most useful for fact-retrieval questions with specific expected answers.

---

## 9. Correct Answers Count (`correct_answers_count`)

### What is it?
A binary classification metric that counts answers classified as **CORRECT**. An answer is correct if it substantially matches the ground truth and addresses the question appropriately.

### Relevance to Evaluation
This metric provides:
- Simple **pass/fail** assessment
- Easy-to-understand aggregate statistics
- Useful for calculating accuracy percentages across datasets
- Foundation for answer classification analysis

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| `answer` | `str` | The generated answer |
| `ground_truth` | `str` | The reference correct answer |
| `question` | `str` | The original question |

### Calculation

**Classification Logic:**

1. **Check for uncertainty phrases** → If present, classify as "DON'T KNOW"
2. **LLM binary classification** → CORRECT or WRONG

**Uncertainty Phrases Detected:**
- "i don't know"
- "i do not know"
- "unknown"
- "not sure"
- "cannot determine"
- "no information"
- "insufficient data"
- "unable to answer"

### Pseudo Code
```python
def compute_answer_classification(
    answer: str,
    ground_truth: str,
    question: str
) -> str:
    
    # Step 1: Check for uncertainty phrases
    uncertainty_phrases = [
        "i don't know", "i do not know", "unknown",
        "not sure", "cannot determine", "no information",
        "insufficient data", "unable to answer"
    ]
    
    answer_lower = answer.lower().strip()
    for phrase in uncertainty_phrases:
        if phrase in answer_lower:
            return "DON'T KNOW"
    
    # Step 2: LLM classification
    classification = await llm_classify_answer(question, answer, ground_truth)
    
    return classification  # "CORRECT" or "WRONG"

def aggregate_correct_count(classifications: List[str]) -> int:
    return sum(1 for c in classifications if c == "CORRECT")
```

### Prompt Used

**Answer Classification Prompt:**
```
You are an answer quality evaluator. Classify the candidate answer as CORRECT or WRONG 
by comparing it to the ground truth answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Candidate Answer: {answer}

Classification Criteria:
- CORRECT: The candidate answer is factually correct and addresses the question similarly to the ground truth
- WRONG: The candidate answer is factually incorrect, incomplete, or doesn't properly address the question

Consider:
1. Factual accuracy compared to ground truth
2. Completeness of the answer
3. Whether the core question is answered

Output only one word: CORRECT or WRONG
```

### Output
- **Type**: `int`
- **Range**: `0` to `N` (number of samples)
- **Interpretation**: Higher count = more correct answers

### Real Example from Llama3_test

**Classification Distribution Across 24 Evaluations**:

| Classification | Count | Percentage |
|----------------|-------|------------|
| CORRECT | 17 | 70.8% |
| WRONG | 6 | 25.0% |
| DON'T KNOW | 1 | 4.2% |

**LLM Prompt Used (Answer Classification)**:
```
You are an answer quality evaluator. Classify the candidate answer as CORRECT or WRONG 
by comparing it to the ground truth answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Candidate Answer: {answer}

Classification Criteria:
- CORRECT: The candidate answer is factually correct and addresses the question 
  similarly to the ground truth
- WRONG: The candidate answer is factually incorrect, incomplete, or doesn't 
  properly address the question

Consider:
1. Factual accuracy compared to ground truth
2. Completeness of the answer
3. Whether the core question is answered

Output only one word: CORRECT or WRONG
```

**Detailed Example**:

**Question ID**: `Novel-73586ddc`  
**Question**: *"...which plant known scientifically as Erica vagans is also referred to by another common name?"*

| Method | Answer | Classification |
|--------|--------|----------------|
| local_search | "Cornish heath" | ✅ CORRECT |
| basic_search | "Cornish heath" | ✅ CORRECT |
| llm_with_context | "Cornish heath" | ✅ CORRECT |

**By Method Across All Questions**:
| Method | Correct | Wrong | Don't Know |
|--------|---------|-------|------------|
| local_search | 6 | 2 | 0 |
| basic_search | 6 | 2 | 0 |
| llm_with_context | 5 | 2 | 1 |

---

## 10. Wrong Answers Count (`wrong_answers_count`)

### What is it?
A binary classification metric that counts answers classified as **WRONG**. An answer is wrong if it contains factual errors, contradicts the ground truth, or fails to address the question.

### Relevance to Evaluation
This metric helps:
- Identify **failure cases** for analysis
- Calculate error rates
- Understand where the RAG system struggles
- Prioritize areas for improvement

### Inputs
Same as Correct Answers Count:
| Input | Type | Description |
|-------|------|-------------|
| `answer` | `str` | The generated answer |
| `ground_truth` | `str` | The reference correct answer |
| `question` | `str` | The original question |

### Calculation

Uses the same classification logic as `correct_answers_count`, but counts "WRONG" classifications.

### Pseudo Code
```python
def aggregate_wrong_count(classifications: List[str]) -> int:
    return sum(1 for c in classifications if c == "WRONG")
```

### Classification Criteria (from LLM prompt)
An answer is classified as **WRONG** when:
- Contains factually incorrect information
- Contradicts the ground truth
- Is incomplete to the point of being misleading
- Doesn't address the actual question asked
- Provides irrelevant information

### Output
- **Type**: `int`
- **Range**: `0` to `N` (number of samples)
- **Interpretation**: Lower count = fewer errors

### Real Example from Llama3_test

**Wrong Answer Examples**:

| Question ID | Question (abbreviated) | Ground Truth | Wrong Answer | Method |
|-------------|----------------------|--------------|--------------|--------|
| Novel-74440a6a | "...who married Princess Frederica of Hanover?" | "Baron Alphonse" | (incorrect name) | llm_with_context |
| Novel-322bb52d | "...King Arthur and Land of Lyonesse..." | (specific details) | (missing details) | basic_search |
| Novel-48edb564 | "...significance of sea and cove for Charles..." | (contextual answer) | (incomplete) | local_search |

**Error Analysis by Question Type**:
| Question Type | Total | Correct | Wrong | Don't Know |
|---------------|-------|---------|-------|------------|
| Fact Retrieval | 6 | 5 | 1 | 0 |
| Complex Reasoning | 6 | 4 | 2 | 0 |
| Contextual Summarize | 6 | 5 | 1 | 0 |
| Creative Generation | 6 | 3 | 2 | 1 |

**Interpretation**: Creative generation and complex reasoning questions have higher error rates, indicating these are more challenging for the RAG system.

---

## 11. Don't Know Answers Count (`dont_know_answers_count`)

### What is it?
A classification metric that counts answers where the system **admitted uncertainty** rather than providing an answer. This includes explicit "I don't know" responses and other uncertainty indicators.

### Relevance to Evaluation
This metric is important because:
- Measures the system's **calibration** - does it know when it doesn't know?
- High count may indicate retrieval failures or knowledge gaps
- Low count with high wrong answers may indicate overconfidence
- Helps identify questions outside the system's knowledge domain

### Inputs
Same as other classification metrics:
| Input | Type | Description |
|-------|------|-------------|
| `answer` | `str` | The generated answer |
| `ground_truth` | `str` | The reference correct answer |
| `question` | `str` | The original question |

### Calculation

**Detection Method:**
Checks for presence of uncertainty phrases in the answer **before** LLM classification.

### Pseudo Code
```python
UNCERTAINTY_PHRASES = [
    "i don't know",
    "i do not know", 
    "unknown",
    "not sure",
    "cannot determine",
    "no information",
    "insufficient data",
    "unable to answer",
    "cannot answer",
    "don't have enough information",
    "not available",
    "no data"
]

def is_dont_know_answer(answer: str) -> bool:
    answer_lower = answer.lower().strip()
    
    for phrase in UNCERTAINTY_PHRASES:
        if phrase in answer_lower:
            return True
    
    # Also check for very short non-answers
    if len(answer_lower) < 10 and any(
        word in answer_lower 
        for word in ["unknown", "n/a", "none", "null"]
    ):
        return True
    
    return False

def aggregate_dont_know_count(classifications: List[str]) -> int:
    return sum(1 for c in classifications if c == "DON'T KNOW")
```

### Output
- **Type**: `int`
- **Range**: `0` to `N` (number of samples)
- **Interpretation**: 
  - Some "don't know" responses are good (honest uncertainty)
  - Too many may indicate retrieval issues
  - Context needed to interpret this metric

### Real Example from Llama3_test

**Don't Know Detection**:

In the Llama3_test evaluation, **1 out of 24 answers** was classified as "Don't Know":

**Question ID**: `Novel-9d9b1ed1`  
**Question**: *"Imagine you are a fish living in the cave near Lizard. Write a diary entry about your encounter with John Curgenven's boat and the launce."*  
**Question Type**: Creative Generation

**Answer (llm_with_context method)**: The answer indicated uncertainty or inability to complete the creative task from the fish's perspective.

**Detection Method**:
```python
UNCERTAINTY_PHRASES = [
    "i don't know", "i do not know", "unknown",
    "not sure", "cannot determine", "no information",
    "insufficient data", "unable to answer"
]

# If any phrase detected in answer → classify as "DON'T KNOW"
```

**Distribution in Llama3_test**:
| Method | Don't Know Count |
|--------|-----------------|
| llm_with_context | 1 |
| basic_search | 0 |
| local_search | 0 |

**Interpretation**: The single "don't know" response came from a creative generation task where the LLM expressed uncertainty. This is relatively rare (4.2% of responses), suggesting the system is generally confident in its answers.
