import re
import numpy as np
import threading
from typing import Any, Dict, Optional, List, Tuple
import logging
from utils.error_handling import safe_metric_computation

# Lazy loading for BERTScore to avoid slow imports
# Thread-safe initialization using double-checked locking pattern
_bert_scorer = None
_bert_scorer_lock = threading.Lock()


def _get_bert_scorer(model_type: str = "microsoft/deberta-xlarge-mnli", device: str = None):
    """Lazy load BERTScore scorer with thread-safe initialization.
    
    Uses double-checked locking to ensure only one instance is created
    even when multiple threads attempt initialization concurrently.
    """
    global _bert_scorer
    
    # Fast path: return existing scorer without lock
    if _bert_scorer is not None:
        return _bert_scorer
    
    # Slow path: acquire lock and initialize
    with _bert_scorer_lock:
        # Double-check after acquiring lock (another thread may have initialized)
        if _bert_scorer is not None:
            return _bert_scorer
        
        try:
            from bert_score import BERTScorer
            import torch
            
            # Auto-detect device if not specified
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # NOTE: rescale_with_baseline=False is intentional!
            # With rescale_with_baseline=True, scores are computed as:
            #   rescaled = (raw - baseline) / (1 - baseline)
            # When comparing long verbose answers to short ground truths,
            # raw scores can fall below the baseline, resulting in negative
            # rescaled scores that get clamped to 0.0, losing all signal.
            # Without rescaling, scores remain in [0,1] and provide useful
            # similarity information even for length-mismatched comparisons.
            _bert_scorer = BERTScorer(
                model_type=model_type,
                device=device,
                rescale_with_baseline=False,
                lang="en"
            )
        except ImportError:
            raise ImportError(
                "bert_score package is required for BERTScore metric. "
                "Install it with: pip install bert_score"
            )
    return _bert_scorer


class TextMetrics:
    """Simple, interpretable text-based evaluation metrics."""
    
    def __init__(self, logger: logging.Logger, bert_model: str = "microsoft/deberta-xlarge-mnli"):
        self.logger = logger
        self.bert_model = bert_model
        self._bert_scorer = None  # Lazy initialization
    

    
    def compute_f1_score(self, answer: str, ground_truth: str) -> float:
        """Compute F1 score between answer and ground truth (0-100 scale)."""
        with safe_metric_computation('f1_score', self.logger, fallback_value=np.nan):
            # Missing ground truth = cannot compute (return np.nan to indicate missing data)
            if not ground_truth:
                return np.nan
            # Empty answer with valid ground truth = legitimate zero score
            if not answer:
                return 0.0
            
            # CRITICAL-004 FIX: For short ground truths (<=3 words), use all words without stop word filtering
            gt_words = ground_truth.lower().split()
            if len(gt_words) <= 3:
                # Short ground truth - use all tokens to avoid filtering everything out
                answer_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
                gt_tokens = set(gt_words)
            else:
                # Longer ground truth - use important words only
                answer_tokens = set(self._extract_important_words(answer))
                gt_tokens = set(self._extract_important_words(ground_truth))
            
            # No tokens in ground truth after processing = cannot compute
            if not gt_tokens:
                self.logger.warning(f"F1 score: ground truth has no tokens after processing: '{ground_truth}'")
                return np.nan
            
            # No tokens in answer = legitimate zero score
            if not answer_tokens:
                return 0.0
            
            # Calculate precision and recall
            intersection = answer_tokens & gt_tokens
            precision = len(intersection) / len(answer_tokens)
            recall = len(intersection) / len(gt_tokens)
            
            # Calculate F1 score
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1 * 100.0
    
    def compute_exact_match_score(self, answer: str, ground_truth: str) -> float:
        """Compute exact string match as a clear binary score (0 or 100)."""
        with safe_metric_computation('exact_match_score', self.logger, fallback_value=np.nan):
            # Missing ground truth = cannot compute (return np.nan to indicate missing data)
            if not ground_truth:
                return np.nan
            # Empty answer with valid ground truth = legitimate zero score
            if not answer:
                return 0.0
            
            # Normalize whitespace and case
            answer_norm = re.sub(r'\s+', ' ', answer.strip().lower())
            gt_norm = re.sub(r'\s+', ' ', ground_truth.strip().lower())
            
            return 100.0 if answer_norm == gt_norm else 0.0
    
    def _extract_important_words(self, text: str) -> set:
        """Extract important words (excluding common stop words)."""
        # Simple stop words list
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
            'to', 'was', 'were', 'will', 'with', 'this', 'these', 'they', 'them'
        }
        
        # Extract words, convert to lowercase, remove stop words
        words = set(re.findall(r'\b\w+\b', text.lower()))
        return words - stop_words
    
    def compute_bert_score(self, answer: str, ground_truth: str) -> Dict[str, float]:
        """
        Compute BERTScore between answer and ground truth.
        
        BERTScore leverages pre-trained contextual embeddings from BERT to compute
        semantic similarity between candidate and reference texts. It computes
        token-level similarity using cosine similarity of BERT embeddings.
        
        Args:
            answer: The candidate/generated answer text
            ground_truth: The reference/ground truth text
            
        Returns:
            Dictionary containing:
                - bert_score_precision: How much of the answer is supported by ground truth (0-100%)
                - bert_score_recall: How much of the ground truth is covered by the answer (0-100%)
                - bert_score_f1: Harmonic mean of precision and recall (0-100%)
        """
        with safe_metric_computation('bert_score', self.logger, fallback_value={
            'bert_score_precision': np.nan,
            'bert_score_recall': np.nan,
            'bert_score_f1': np.nan
        }):
            # Handle edge cases
            if not answer and not ground_truth:
                return {
                    'bert_score_precision': 100.0,
                    'bert_score_recall': 100.0,
                    'bert_score_f1': 100.0
                }
            
            if not answer or not ground_truth:
                return {
                    'bert_score_precision': 0.0,
                    'bert_score_recall': 0.0,
                    'bert_score_f1': 0.0
                }
            
            try:
                # Get the BERTScore scorer (lazy loaded)
                scorer = _get_bert_scorer(self.bert_model)
                
                # Compute BERTScore
                # Note: BERTScorer expects lists of strings
                P, R, F1 = scorer.score([answer], [ground_truth])
                
                # Convert to percentages (0-100 scale) and extract scalar values
                precision = float(P[0].item()) * 100.0
                recall = float(R[0].item()) * 100.0
                f1 = float(F1[0].item()) * 100.0
                
                # Clamp values to valid range (rescaled scores can occasionally exceed bounds)
                precision = max(0.0, min(100.0, precision))
                recall = max(0.0, min(100.0, recall))
                f1 = max(0.0, min(100.0, f1))
                
                return {
                    'bert_score_precision': precision,
                    'bert_score_recall': recall,
                    'bert_score_f1': f1
                }
                
            except ImportError as e:
                self.logger.warning(f"BERTScore not available: {e}")
                return {
                    'bert_score_precision': np.nan,
                    'bert_score_recall': np.nan,
                    'bert_score_f1': np.nan
                }
            except Exception as e:
                self.logger.error(f"Error computing BERTScore: {e}")
                return {
                    'bert_score_precision': np.nan,
                    'bert_score_recall': np.nan,
                    'bert_score_f1': np.nan
                }
    
    def compute_bert_score_f1(self, answer: str, ground_truth: str) -> float:
        """
        Compute BERTScore with adaptive score selection based on length ratio.
        
        When answer length differs significantly from ground truth, F1 score can be 
        misleading. This method adapts the returned score:
        - If answer is 5x+ longer than ground truth: use recall (measures if GT content is found)
        - If ground truth is 5x+ longer than answer: use precision (measures if answer is relevant)
        - Otherwise: use F1 (balanced measure)
        
        Args:
            answer: The candidate/generated answer text
            ground_truth: The reference/ground truth text
            
        Returns:
            Adaptive BERTScore (0-100%) - may be recall, precision, or F1 based on length ratio
        """
        scores = self.compute_bert_score(answer, ground_truth)
        
        # Calculate length ratio to determine which score to use
        answer_len = len(answer.split()) if answer else 0
        gt_len = len(ground_truth.split()) if ground_truth else 0
        
        # Avoid division by zero
        if gt_len == 0 or answer_len == 0:
            return scores.get('bert_score_f1', np.nan)
        
        length_ratio = answer_len / gt_len
        
        # For very verbose answers (5x+ longer), recall is more appropriate
        # because it measures how much of the ground truth is covered
        if length_ratio >= 5.0:
            self.logger.debug(
                f"Using BERTScore recall (length ratio={length_ratio:.1f}x): "
                f"answer has {answer_len} words, ground_truth has {gt_len} words"
            )
            return scores.get('bert_score_recall', np.nan)
        
        # For very terse answers (ground truth 5x+ longer), precision is more appropriate
        # because it measures how relevant the short answer is
        if length_ratio <= 0.2:
            self.logger.debug(
                f"Using BERTScore precision (length ratio={length_ratio:.1f}x): "
                f"answer has {answer_len} words, ground_truth has {gt_len} words"
            )
            return scores.get('bert_score_precision', np.nan)
        
        # For similar lengths, F1 is the appropriate balanced measure
        return scores.get('bert_score_f1', np.nan)