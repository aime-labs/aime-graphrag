from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import yaml
import os
from pathlib import Path


@dataclass
class EvaluationConfig:
    """Centralized evaluation configuration."""
    
    # Model configuration
    llm_model: str
    embedding_model: str
    methods: List[str]
    output_dir: str
    
    # LLM-as-a-judge configuration
    judge_llm_model: Optional[str] = None  # If None, uses llm_model
    judge_api_base_url: Optional[str] = None  # If None, uses api_base_url
    judge_api_key: Optional[str] = None  # If None, uses api_key
    
    # Optional model configuration
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    
    # Evaluation settings
    metrics: Optional[List[str]] = None
    max_samples_per_type: Optional[int] = None
    question_types_filter: Optional[List[str]] = None  # Filter to specific question types
    pairwise_evaluation: bool = False  # Enable pairwise A/B testing
    
    # LLM parameters for reproducible benchmarking
    # CRITICAL: These ensure consistent behavior across all LLM adapters for fair comparison
    llm_temperature: float = 0.0  # Use 0.0 for deterministic outputs during benchmarking
    llm_max_tokens: int = 2500    # Maximum tokens for generated responses
    llm_top_p: float = 1.0        # Nucleus sampling (1.0 = no filtering for deterministic mode)
    llm_top_k: int = 40           # Top-k sampling parameter
    reproducible_mode: bool = True  # When True, forces temperature=0 for reproducibility
    
    # Context size limits for llm_with_context method
    # These prevent API payload errors with large documents
    # AIME API has 120K token limit (~480K chars)
    # We use conservative limits to leave room for prompt and response
    max_llm_context_size: int = 60000   # Total max chars for all docs (~15K tokens)
    max_single_doc_size: int = 40000    # Max chars per individual document (~10K tokens)
    
    # Output settings
    results_file: str = "results.json"
    metrics_file: str = "metrics.json"
    logs_file: str = "logs.json"
    query_log_file: str = "query.json"
    
    # Judge logging settings (controls detailed judge.json output)
    enable_judge_logging: bool = True  # Set to False to disable detailed judge interaction logging
    judge_log_file: str = "judge.json"
    
    # Performance settings
    max_concurrent_tasks: int = 10
    batch_size: int = 2
    
    # Project settings
    project_path: Optional[str] = None
    config_path: Optional[str] = None
    questions_path: Optional[str] = None
    input_json_path: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'EvaluationConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_args(cls, args) -> 'EvaluationConfig':
        """Create configuration from command-line arguments."""
        return cls(
            llm_model=getattr(args, 'model', 'gpt-4-turbo'),
            embedding_model=getattr(args, 'embedding_model', 'BAAI/bge-large-en-v1.5'),
            api_base_url=getattr(args, 'base_url', 'https://api.openai.com/v1'),
            judge_llm_model=getattr(args, 'judge_model', 'llama4_chat'),  # Default to llama4_chat for judge (also supports mistral_chat, gpt_oss_chat)
            judge_api_base_url=getattr(args, 'judge_base_url', None),  # Will default to main api_base_url if None
            judge_api_key=getattr(args, 'judge_api_key', None),  # Will default to main api_key if None
            methods=getattr(args, 'methods', ['local_search', 'global_search', 'basic_search', 'direct_llm_with_evidence']),
            output_dir=getattr(args, 'output_dir', './output'),
            max_samples_per_type=getattr(args, 'num_samples', None),
            project_path=getattr(args, 'project_path', None),
            config_path=getattr(args, 'config', None),
            questions_path=getattr(args, 'questions_json', None),
            input_json_path=getattr(args, 'input_json', None),
            pairwise_evaluation=getattr(args, 'pairwise', False),
            # LLM parameters for reproducible benchmarking
            llm_temperature=getattr(args, 'llm_temperature', 0.0),
            llm_max_tokens=getattr(args, 'llm_max_tokens', 2500),
            llm_top_p=getattr(args, 'llm_top_p', 1.0),
            reproducible_mode=getattr(args, 'reproducible_mode', True)
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.llm_model:
            errors.append("LLM model must be specified")
        
        if not self.embedding_model:
            errors.append("Embedding model must be specified")
        
        if not self.methods:
            errors.append("At least one method must be specified")
        
        if not self.output_dir:
            errors.append("Output directory must be specified")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        return errors
    
    def get_results_path(self) -> str:
        """Get full path to results file."""
        return os.path.join(self.output_dir, self.results_file)
    
    def get_metrics_path(self) -> str:
        """Get full path to metrics file."""
        return os.path.join(self.output_dir, self.metrics_file)
    
    def get_logs_path(self) -> str:
        """Get full path to logs file."""
        return os.path.join(self.output_dir, self.logs_file)
    
    def get_query_log_path(self) -> str:
        """Get full path to query log file."""
        return os.path.join(self.output_dir, self.query_log_file)
    
    def get_judge_log_path(self) -> str:
        """Get full path to judge log file."""
        return os.path.join(self.output_dir, self.judge_log_file) 