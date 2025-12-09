from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from enum import Enum


class Architecture(Enum):
    """Supported system architectures."""
    GRAPHRAG = "graphrag"
    RAG = "rag"
    DIRECT_LLM = "direct_llm"
    ALL = "all"


class QuestionType(Enum):
    """Supported question types."""
    FACT_RETRIEVAL = "Fact Retrieval"
    COMPLEX_REASONING = "Complex Reasoning"
    CONTEXTUAL_SUMMARIZE = "Contextual Summarize"
    CREATIVE_GENERATION = "Creative Generation"
    RETRIEVAL = "Retrieval"
    ALL = "all"


@dataclass
class MetricConfig:
    """Configuration for a single metric."""
    name: str
    description: str
    question_types: List[str]
    architectures: List[str]
    required: bool = False
    async_computation: bool = True
    fallback_value: Optional[float] = None
    category: str = "general"  # Categories: ragas, text, semantic, retrieval, hallucination


class MetricRegistry:
    """Registry for managing evaluation metrics."""
    
    def __init__(self):
        self.metrics = self._initialize_metrics()
    
    def _initialize_metrics(self) -> Dict[str, MetricConfig]:
        """Initialize all available metrics."""
        return {
            # =================================================================
            # RAGAS METRICS (LLM-as-a-Judge, Primary Evaluation Metrics)
            # Reference: RAGAS - Retrieval Augmented Generation Assessment
            # =================================================================
            'ragas_faithfulness': MetricConfig(
                name='ragas_faithfulness',
                description='RAGAS Faithfulness: Measures factual consistency of answer with retrieved context (0-100%)',
                question_types=[QuestionType.ALL.value],
                architectures=[Architecture.GRAPHRAG.value, Architecture.RAG.value],
                required=True,
                fallback_value=0.0,
                category='ragas'
            ),
            'ragas_context_precision': MetricConfig(
                name='ragas_context_precision',
                description='RAGAS Context Precision: Measures if retrieved contexts are relevant (0-100%)',
                question_types=[QuestionType.ALL.value],
                architectures=[Architecture.GRAPHRAG.value, Architecture.RAG.value],
                required=True,
                fallback_value=0.0,
                category='ragas'
            ),
            'ragas_context_recall': MetricConfig(
                name='ragas_context_recall',
                description='RAGAS Context Recall: Measures if contexts contain all needed info (0-100%)',
                question_types=[QuestionType.ALL.value],
                architectures=[Architecture.GRAPHRAG.value, Architecture.RAG.value],
                required=True,
                fallback_value=0.0,
                category='ragas'
            ),
            'ragas_answer_relevance': MetricConfig(
                name='ragas_answer_relevance',
                description='RAGAS Answer Relevance: Measures how well answer addresses the question (0-100%)',
                question_types=[QuestionType.ALL.value],
                architectures=[Architecture.ALL.value],
                required=True,
                fallback_value=0.0,
                category='ragas'
            ),
            'ragas_score': MetricConfig(
                name='ragas_score',
                description='RAGAS Composite Score: Weighted average of all RAGAS metrics (0-100%)',
                question_types=[QuestionType.ALL.value],
                architectures=[Architecture.ALL.value],
                required=False,
                fallback_value=0.0,
                category='ragas'
            ),
            
            # =================================================================
            # SEMANTIC METRICS (LLM-based Answer Quality)
            # =================================================================
            'factual_accuracy_grade': MetricConfig(
                name='factual_accuracy_grade',
                description='Factual accuracy grade using GEval (A/B/C/D/E letter grade). Not applicable to creative generation tasks.',
                # Exclude Creative Generation since factual accuracy is not meaningful for creative tasks
                question_types=[
                    QuestionType.FACT_RETRIEVAL.value,
                    QuestionType.COMPLEX_REASONING.value,
                    QuestionType.CONTEXTUAL_SUMMARIZE.value,
                    QuestionType.RETRIEVAL.value
                ],
                architectures=[Architecture.ALL.value],
                required=True,
                fallback_value="N/A",  # N/A for skipped metrics (e.g., creative tasks)
                category='semantic'
            ),
            'semantic_similarity_percentage': MetricConfig(
                name='semantic_similarity_percentage',
                description='Embedding-based semantic similarity (0-100%)',
                question_types=[QuestionType.ALL.value],
                architectures=[Architecture.ALL.value],
                required=True,
                fallback_value=0.0,
                category='semantic'
            ),
            
            # =================================================================
            # TEXT-BASED METRICS (BERTScore)
            # =================================================================
            'bert_score_f1': MetricConfig(
                name='bert_score_f1',
                description='BERTScore F1: Semantic similarity using BERT embeddings (0-100%)',
                question_types=[QuestionType.ALL.value],
                architectures=[Architecture.ALL.value],
                required=True,
                fallback_value=0.0,
                category='text'
            ),
            
            # =================================================================
            # ANSWER CLASSIFICATION METRICS (Correct/Wrong/Don't Know)
            # =================================================================
            'correct_answers_count': MetricConfig(
                name='correct_answers_count',
                description='Count of correct answers (0 or 1 per question)',
                question_types=[QuestionType.ALL.value],
                architectures=[Architecture.ALL.value],
                required=True,
                fallback_value=0.0,
                category='classification'
            ),
            'wrong_answers_count': MetricConfig(
                name='wrong_answers_count',
                description='Count of wrong answers (0 or 1 per question)',
                question_types=[QuestionType.ALL.value],
                architectures=[Architecture.ALL.value],
                required=True,
                fallback_value=0.0,
                category='classification'
            ),
            'dont_know_answers_count': MetricConfig(
                name='dont_know_answers_count',
                description='Count of "don\'t know" answers (0 or 1 per question)',
                question_types=[QuestionType.ALL.value],
                architectures=[Architecture.ALL.value],
                required=True,
                fallback_value=0.0,
                category='classification'
            ),
        }
    
    def get_metrics_for_config(self, question_type: str, architecture: str) -> List[str]:
        """Get appropriate metrics for given question type and architecture."""
        metrics = []
        
        for metric_name, metric_config in self.metrics.items():
            # Check if metric applies to this question type
            type_match = (
                question_type in metric_config.question_types or 
                QuestionType.ALL.value in metric_config.question_types
            )
            
            # Check if metric applies to this architecture
            arch_match = (
                architecture in metric_config.architectures or 
                Architecture.ALL.value in metric_config.architectures
            )
            
            if type_match and arch_match:
                metrics.append(metric_name)
        
        return metrics
    
    def get_required_metrics(self, question_type: str, architecture: str) -> List[str]:
        """Get required metrics for given question type and architecture."""
        all_metrics = self.get_metrics_for_config(question_type, architecture)
        return [
            metric for metric in all_metrics 
            if self.metrics[metric].required
        ]
    
    def get_metrics_by_category(self, category: str) -> List[str]:
        """Get all metrics in a specific category."""
        return [
            name for name, config in self.metrics.items()
            if config.category == category
        ]
    
    def get_ragas_metrics(self) -> List[str]:
        """Get all RAGAS metrics."""
        return self.get_metrics_by_category('ragas')
    
    def get_metric_config(self, metric_name: str) -> Optional[MetricConfig]:
        """Get configuration for a specific metric."""
        return self.metrics.get(metric_name)
    
    def get_all_metrics(self) -> Dict[str, MetricConfig]:
        """Get all available metrics."""
        return self.metrics.copy()
    
    def validate_metrics(self, metrics: List[str]) -> List[str]:
        """Validate that all requested metrics exist."""
        errors = []
        for metric in metrics:
            if metric not in self.metrics:
                errors.append(f"Unknown metric: {metric}")
        return errors 