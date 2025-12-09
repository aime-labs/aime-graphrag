import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from utils.error_handling import safe_metric_computation, parse_json_response

class PairwiseMetrics:
    """
    Pairwise Evaluation Metrics (A/B Testing).
    
    Compares two answers to the same question and determines which one is better
    using an LLM judge.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.prompts_dir = Path(__file__).parent.parent / "prompt_templates"
        
    def _load_prompt(self, template_name: str, **kwargs) -> str:
        """Load and format a prompt template from file."""
        try:
            template_path = self.prompts_dir / f"{template_name}.txt"
            
            if not template_path.exists():
                self.logger.warning(f"Prompt template not found: {template_path}")
                return ""
                
            with open(template_path, "r") as f:
                template = f.read()
                
            return template.format(**kwargs)
        except Exception as e:
            self.logger.error(f"Error loading prompt template {template_name}: {e}")
            return ""

    async def compare_answers(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        ground_truth: Optional[str] = None,
        llm_adapter: Any = None,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Compare two answers and determine which is better.
        
        Args:
            question: The input question
            answer_a: First answer
            answer_b: Second answer
            ground_truth: Optional reference answer
            llm_adapter: LLM adapter for API calls
            max_retries: Maximum retry attempts
            
        Returns:
            Dictionary with comparison results:
            {
                "winner": "A", "B", or "Tie",
                "reasoning": "explanation",
                "score_a": 0-10,
                "score_b": 0-10
            }
        """
        with safe_metric_computation('pairwise_comparison', self.logger, fallback_value={"winner": "Error"}):
            prompt = self._load_prompt(
                "pairwise_comparison",
                question=question,
                answer_a=answer_a,
                answer_b=answer_b,
                ground_truth=ground_truth if ground_truth else "Not provided"
            )
            
            for attempt in range(max_retries + 1):
                try:
                    response = await llm_adapter.ainvoke(prompt)
                    result = parse_json_response(
                        response.content,
                        self.logger,
                        "pairwise comparison"
                    )
                    
                    if isinstance(result, dict) and "winner" in result:
                        return result
                        
                except Exception as e:
                    if attempt == max_retries:
                        self.logger.warning(f"Pairwise comparison failed: {e}")
                    continue
            
            return {"winner": "Error", "reasoning": "All attempts failed"}
