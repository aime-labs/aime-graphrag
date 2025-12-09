"""
Judge LLM Interaction Tracker

Tracks all interactions with the judge LLM during evaluation for transparency,
debugging, and analysis purposes. Results are saved to judge.json.
"""

import time
import hashlib
import asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
import logging


@dataclass
class JudgeInteraction:
    """Single judge LLM interaction record."""
    
    # Identification
    interaction_id: str  # Unique ID for this interaction
    question_id: str  # ID of the question being evaluated
    metric_name: str  # Name of the metric being computed (e.g., 'factual_accuracy', 'ragas_faithfulness')
    
    # Request details
    prompt: str  # The full prompt sent to the judge LLM
    prompt_template: Optional[str] = None  # Name/type of prompt template used
    
    # Input context
    question: str = ""  # The original question
    answer: str = ""  # The answer being evaluated
    ground_truth: str = ""  # The ground truth/reference answer
    contexts: List[str] = field(default_factory=list)  # Retrieved contexts if applicable
    
    # Response details
    raw_response: str = ""  # Raw response from the judge LLM
    parsed_result: Optional[Dict[str, Any]] = None  # Parsed/structured result
    
    # Computed values
    score: Optional[float] = None  # The final score computed
    classification: Optional[str] = None  # Classification result if applicable
    reasoning: str = ""  # Reasoning/explanation from the judge
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0  # Time taken for the LLM call
    success: bool = True  # Whether the call succeeded
    error_message: str = ""  # Error message if failed
    retry_count: int = 0  # Number of retries needed
    
    # Model info
    model_name: str = ""  # Name of the judge model used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class JudgeTracker:
    """
    Tracks all judge LLM interactions during evaluation.
    
    Usage:
        tracker = JudgeTracker(logger)
        
        # Start tracking an interaction
        interaction_id = tracker.start_interaction(
            question_id="Q123",
            metric_name="factual_accuracy",
            prompt=prompt_text,
            question=question,
            answer=answer,
            ground_truth=ground_truth
        )
        
        # Record the response
        tracker.record_response(
            interaction_id=interaction_id,
            raw_response=response.content,
            parsed_result={"score": 85},
            score=85.0,
            reasoning="Answer is mostly correct..."
        )
        
        # Or record an error
        tracker.record_error(
            interaction_id=interaction_id,
            error_message="LLM timeout"
        )
        
        # Get all interactions
        all_interactions = tracker.get_all_interactions()
    """
    
    # Default maximum interactions before memory cleanup (can be overridden)
    DEFAULT_MAX_INTERACTIONS = 10000
    
    def __init__(self, logger: logging.Logger, max_interactions: int = None):
        """Initialize the JudgeTracker.
        
        Args:
            logger: Logger instance for debug/warning messages
            max_interactions: Maximum number of interactions to keep in memory.
                            When exceeded, oldest interactions are removed.
                            Set to None or 0 for unlimited (not recommended for large evaluations).
                            Default is 10000 interactions.
        """
        self.logger = logger
        self.interactions: Dict[str, JudgeInteraction] = {}
        self._interaction_list: List[JudgeInteraction] = []
        self._max_interactions = max_interactions if max_interactions is not None else self.DEFAULT_MAX_INTERACTIONS
        self._eviction_count = 0  # Track how many interactions have been evicted
    
    def _evict_oldest_interactions(self, count: int = None):
        """Remove oldest interactions to free memory.
        
        Args:
            count: Number of interactions to remove. If None, removes 10% of max.
        """
        if not self._interaction_list:
            return
        
        if count is None:
            # Remove 10% to avoid frequent evictions
            count = max(1, self._max_interactions // 10)
        
        count = min(count, len(self._interaction_list))
        
        # Remove oldest interactions
        evicted = self._interaction_list[:count]
        self._interaction_list = self._interaction_list[count:]
        
        # Remove from dict as well
        for interaction in evicted:
            self.interactions.pop(interaction.interaction_id, None)
        
        self._eviction_count += len(evicted)
        self.logger.debug(f"Evicted {len(evicted)} oldest judge interactions (total evicted: {self._eviction_count})")
    
    def _check_memory_limit(self):
        """Check if memory limit is reached and evict if necessary."""
        if self._max_interactions > 0 and len(self._interaction_list) >= self._max_interactions:
            self._evict_oldest_interactions()
    
    def _generate_interaction_id(self, question_id: str, metric_name: str) -> str:
        """Generate a unique interaction ID."""
        timestamp = str(time.time())
        unique_str = f"{question_id}_{metric_name}_{timestamp}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]
    
    def start_interaction(
        self,
        question_id: str,
        metric_name: str,
        prompt: str,
        question: str = "",
        answer: str = "",
        ground_truth: str = "",
        contexts: Optional[List[str]] = None,
        prompt_template: Optional[str] = None,
        model_name: str = ""
    ) -> str:
        """
        Start tracking a new judge interaction.
        
        Returns:
            interaction_id: Unique ID to reference this interaction later
        """
        # Check memory limit before adding new interaction
        self._check_memory_limit()
        
        interaction_id = self._generate_interaction_id(question_id, metric_name)
        
        interaction = JudgeInteraction(
            interaction_id=interaction_id,
            question_id=question_id,
            metric_name=metric_name,
            prompt=prompt,
            prompt_template=prompt_template,
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            contexts=contexts or [],
            timestamp=time.time(),
            model_name=model_name
        )
        
        self.interactions[interaction_id] = interaction
        self._interaction_list.append(interaction)
        
        self.logger.debug(f"Started judge interaction {interaction_id} for metric {metric_name}")
        return interaction_id
    
    def record_response(
        self,
        interaction_id: str,
        raw_response: str,
        parsed_result: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None,
        classification: Optional[str] = None,
        reasoning: str = "",
        duration_ms: float = 0.0,
        retry_count: int = 0
    ):
        """Record the successful response from the judge LLM."""
        if interaction_id not in self.interactions:
            self.logger.warning(f"Unknown interaction ID: {interaction_id}")
            return
        
        interaction = self.interactions[interaction_id]
        interaction.raw_response = raw_response
        interaction.parsed_result = parsed_result
        interaction.score = score
        interaction.classification = classification
        interaction.reasoning = reasoning
        interaction.duration_ms = duration_ms
        interaction.success = True
        interaction.retry_count = retry_count
        
        self.logger.debug(f"Recorded response for interaction {interaction_id}: score={score}")
    
    def record_error(
        self,
        interaction_id: str,
        error_message: str,
        raw_response: str = "",
        duration_ms: float = 0.0,
        retry_count: int = 0
    ):
        """Record an error for a judge interaction."""
        if interaction_id not in self.interactions:
            self.logger.warning(f"Unknown interaction ID: {interaction_id}")
            return
        
        interaction = self.interactions[interaction_id]
        interaction.raw_response = raw_response
        interaction.error_message = error_message
        interaction.success = False
        interaction.duration_ms = duration_ms
        interaction.retry_count = retry_count
        
        self.logger.debug(f"Recorded error for interaction {interaction_id}: {error_message}")
    
    def get_all_interactions(self) -> List[Dict[str, Any]]:
        """Get all interactions as a list of dictionaries."""
        return [interaction.to_dict() for interaction in self._interaction_list]
    
    def get_interactions_by_question(self, question_id: str) -> List[Dict[str, Any]]:
        """Get all interactions for a specific question."""
        return [
            interaction.to_dict() 
            for interaction in self._interaction_list 
            if interaction.question_id == question_id
        ]
    
    def get_interactions_by_metric(self, metric_name: str) -> List[Dict[str, Any]]:
        """Get all interactions for a specific metric."""
        return [
            interaction.to_dict() 
            for interaction in self._interaction_list 
            if interaction.metric_name == metric_name
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all judge interactions."""
        total = len(self._interaction_list)
        successful = sum(1 for i in self._interaction_list if i.success)
        failed = total - successful
        
        # Group by metric
        by_metric: Dict[str, Dict[str, Any]] = {}
        for interaction in self._interaction_list:
            metric = interaction.metric_name
            if metric not in by_metric:
                by_metric[metric] = {
                    "count": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_duration_ms": 0.0,
                    "scores": []
                }
            by_metric[metric]["count"] += 1
            if interaction.success:
                by_metric[metric]["successful"] += 1
                if interaction.score is not None:
                    by_metric[metric]["scores"].append(interaction.score)
            else:
                by_metric[metric]["failed"] += 1
            by_metric[metric]["total_duration_ms"] += interaction.duration_ms
        
        # Calculate averages
        for metric in by_metric:
            scores = by_metric[metric]["scores"]
            if scores:
                by_metric[metric]["avg_score"] = sum(scores) / len(scores)
            else:
                by_metric[metric]["avg_score"] = None
            by_metric[metric]["avg_duration_ms"] = (
                by_metric[metric]["total_duration_ms"] / by_metric[metric]["count"]
                if by_metric[metric]["count"] > 0 else 0
            )
            # Remove the raw scores list to keep summary concise
            del by_metric[metric]["scores"]
        
        return {
            "total_interactions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "evicted_interactions": self._eviction_count,  # Track memory management
            "max_interactions_limit": self._max_interactions,
            "by_metric": by_metric
        }
    
    def clear(self):
        """Clear all tracked interactions."""
        self.interactions.clear()
        self._interaction_list.clear()
        self._eviction_count = 0
        self.logger.info("Judge tracker cleared")


class TrackedLLMAdapter:
    """
    Wrapper for LLM adapters that automatically tracks all interactions.
    
    This provides a transparent way to log all judge LLM calls without
    modifying the underlying metric computation code.
    
    Usage:
        tracked_adapter = TrackedLLMAdapter(
            llm_adapter=judge_adapter,
            tracker=judge_tracker,
            question_id="Q123",
            default_metric="unknown"
        )
        
        # Use like a normal adapter - interactions are automatically tracked
        response = await tracked_adapter.ainvoke(prompt)
    """
    
    def __init__(
        self,
        llm_adapter: Any,
        tracker: JudgeTracker,
        question_id: str = "",
        question: str = "",
        answer: str = "",
        ground_truth: str = "",
        contexts: Optional[List[str]] = None,
        default_metric: str = "unknown",
        model_name: str = ""
    ):
        self._adapter = llm_adapter
        self._tracker = tracker
        self._question_id = question_id
        self._question = question
        self._answer = answer
        self._ground_truth = ground_truth
        self._contexts = contexts or []
        self._default_metric = default_metric
        self._model_name = model_name
        self._current_metric = default_metric
        self._last_interaction_id: Optional[str] = None  # Track last interaction for updates
    
    def set_metric_context(
        self,
        metric_name: str,
        question: str = "",
        answer: str = "",
        ground_truth: str = "",
        contexts: Optional[List[str]] = None
    ):
        """Update the context for subsequent calls."""
        self._current_metric = metric_name
        if question:
            self._question = question
        if answer:
            self._answer = answer
        if ground_truth:
            self._ground_truth = ground_truth
        if contexts is not None:
            self._contexts = contexts
    
    async def ainvoke(self, prompt: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Invoke LLM and track the interaction."""
        start_time = time.time()
        
        # Start tracking
        interaction_id = self._tracker.start_interaction(
            question_id=self._question_id,
            metric_name=self._current_metric,
            prompt=prompt,
            question=self._question,
            answer=self._answer,
            ground_truth=self._ground_truth,
            contexts=self._contexts,
            model_name=self._model_name
        )
        
        # Store interaction_id for later update with parsed results
        self._last_interaction_id = interaction_id
        
        try:
            # Make the actual call - pass config as keyword argument for compatibility
            if config:
                response = await self._adapter.ainvoke(prompt, config=config)
            else:
                response = await self._adapter.ainvoke(prompt)
            
            duration_ms = (time.time() - start_time) * 1000
            raw_response = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse the response and extract structured data
            parsed_result = None
            try:
                from utils.error_handling import parse_json_response
                import logging
                # Use a simple logger for parsing - don't fail if parsing fails
                parsed_result = parse_json_response(
                    raw_response, 
                    logging.getLogger(__name__),
                    context=f"judge_tracker_{self._current_metric}",
                    raise_on_failure=False
                )
            except Exception:
                # Parsing failed - that's okay, we'll still record raw response
                pass
            
            # Record successful response with parsed result
            self._tracker.record_response(
                interaction_id=interaction_id,
                raw_response=raw_response,
                parsed_result=parsed_result if isinstance(parsed_result, (dict, list)) else None,
                duration_ms=duration_ms
            )
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            # Record error
            self._tracker.record_error(
                interaction_id=interaction_id,
                error_message=str(e),
                duration_ms=duration_ms
            )
            
            raise
    
    def update_last_interaction(
        self,
        score: Optional[float] = None,
        classification: Optional[str] = None,
        reasoning: str = "",
        parsed_result: Optional[Dict[str, Any]] = None
    ):
        """
        Update the last interaction with additional parsed information.
        
        This method allows metric functions to add score/classification/reasoning
        after they've parsed the LLM response.
        """
        if hasattr(self, '_last_interaction_id') and self._last_interaction_id:
            interaction = self._tracker.interactions.get(self._last_interaction_id)
            if interaction:
                if score is not None:
                    interaction.score = score
                if classification is not None:
                    interaction.classification = classification
                if reasoning:
                    interaction.reasoning = reasoning
                if parsed_result is not None:
                    interaction.parsed_result = parsed_result
    
    async def achat(self, prompt: str) -> Any:
        """Chat with LLM and track the interaction."""
        return await self.ainvoke(prompt)
    
    # Forward any other attribute access to the underlying adapter
    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)


# Global instance for easy access (optional pattern)
_global_tracker: Optional[JudgeTracker] = None


def get_global_tracker() -> Optional[JudgeTracker]:
    """Get the global judge tracker instance."""
    return _global_tracker


def set_global_tracker(tracker: JudgeTracker):
    """Set the global judge tracker instance."""
    global _global_tracker
    _global_tracker = tracker
