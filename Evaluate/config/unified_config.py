"""
Unified configuration system with environment support and validation.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from dotenv import load_dotenv
import jsonschema

from interfaces.core_interfaces import ConfigurationInterface


class Environment(Enum):
    """Supported environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    type: str
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    timeout: Optional[int] = None


@dataclass
class EvaluationSettings:
    """Evaluation-specific settings."""
    methods: List[str] = field(default_factory=lambda: ['local_search', 'global_search'])
    max_samples_per_type: Optional[int] = None
    max_concurrent_tasks: int = 10
    batch_size: int = 5
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds


@dataclass
class ReproducibilityConfig:
    """Reproducibility configuration."""
    random_seed: int = 42
    evaluation_order_seed: int = 123
    enable_deterministic_ordering: bool = True
    track_versions: bool = True
    mlflow_tracking_uri: Optional[str] = None
    experiment_name: Optional[str] = None


@dataclass
class ResourceConfig:
    """Resource management configuration."""
    temp_dir: Optional[str] = None
    max_memory_usage: Optional[str] = None  # e.g., "8GB"
    cleanup_temp_files: bool = True
    connection_pool_size: int = 10
    request_timeout: int = 30


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    enable_structured_logging: bool = True
    enable_metrics: bool = True


class UnifiedConfiguration(ConfigurationInterface):
    """Unified configuration management with environment support."""
    
    # Configuration schema for validation
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "environment": {"type": "string", "enum": ["development", "testing", "staging", "production"]},
            "llm_model": {"$ref": "#/definitions/model_config"},
            "embedding_model": {"$ref": "#/definitions/model_config"},
            "judge_llm_model": {"$ref": "#/definitions/model_config"},
            "reranker": {"$ref": "#/definitions/reranker_config"},
            "evaluation": {"$ref": "#/definitions/evaluation_settings"},
            "reproducibility": {"$ref": "#/definitions/reproducibility_config"},
            "resources": {"$ref": "#/definitions/resource_config"},
            "logging": {"$ref": "#/definitions/logging_config"},
            "project_path": {"type": "string"},
            "output_dir": {"type": "string"},
        },
        "required": ["environment", "llm_model", "embedding_model"],
        "definitions": {
            "model_config": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "api_base_url": {"type": ["string", "null"]},
                    "api_key": {"type": ["string", "null"]},
                    "max_tokens": {"type": ["integer", "null"]},
                    "temperature": {"type": ["number", "null"]},
                    "timeout": {"type": ["integer", "null"]}
                },
                "required": ["name", "type"]
            },
            "reranker_config": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "type": {"type": "string", "enum": ["bge", "cross-encoder", "cohere", "none"]},
                    "model_name": {"type": ["string", "null"]},
                    "device": {"type": "string", "enum": ["cuda", "cpu", "mps"]},
                    "max_length": {"type": "integer", "minimum": 128}
                }
            },
            "evaluation_settings": {
                "type": "object",
                "properties": {
                    "methods": {"type": "array", "items": {"type": "string"}},
                    "max_samples_per_type": {"type": ["integer", "null"]},
                    "max_concurrent_tasks": {"type": "integer", "minimum": 1},
                    "batch_size": {"type": "integer", "minimum": 1},
                    "enable_caching": {"type": "boolean"},
                    "cache_ttl": {"type": "integer", "minimum": 0}
                }
            },
            "reproducibility_config": {
                "type": "object",
                "properties": {
                    "random_seed": {"type": "integer"},
                    "evaluation_order_seed": {"type": "integer"},
                    "enable_deterministic_ordering": {"type": "boolean"},
                    "track_versions": {"type": "boolean"},
                    "mlflow_tracking_uri": {"type": ["string", "null"]},
                    "experiment_name": {"type": ["string", "null"]}
                }
            },
            "resource_config": {
                "type": "object",
                "properties": {
                    "temp_dir": {"type": ["string", "null"]},
                    "max_memory_usage": {"type": ["string", "null"]},
                    "cleanup_temp_files": {"type": "boolean"},
                    "connection_pool_size": {"type": "integer", "minimum": 1},
                    "request_timeout": {"type": "integer", "minimum": 1}
                }
            },
            "logging_config": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                    "format": {"type": "string"},
                    "file_path": {"type": ["string", "null"]},
                    "enable_structured_logging": {"type": "boolean"},
                    "enable_metrics": {"type": "boolean"}
                }
            }
        }
    }
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        self._config: Dict[str, Any] = {}
        self._load_environment_variables()
    
    def _load_environment_variables(self):
        """Load environment variables from .env files."""
        # Load base .env file
        load_dotenv()
        
        # Load environment-specific .env file
        env_file = f".env.{self.environment.value}"
        if os.path.exists(env_file):
            load_dotenv(env_file, override=True)
    
    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            # Merge with existing config
            self._config.update(config_data)
            
            # Override with environment variables
            self._override_with_env_vars()
            
            self.logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
            raise
    
    def _override_with_env_vars(self):
        """Override configuration with environment variables."""
        env_mappings = {
            'EVAL_LLM_MODEL_NAME': 'llm_model.name',
            'EVAL_LLM_MODEL_API_KEY': 'llm_model.api_key',
            'EVAL_LLM_MODEL_BASE_URL': 'llm_model.api_base_url',
            'EVAL_EMBEDDING_MODEL_NAME': 'embedding_model.name',
            'EVAL_EMBEDDING_MODEL_API_KEY': 'embedding_model.api_key',
            'EVAL_MAX_CONCURRENT_TASKS': 'evaluation.max_concurrent_tasks',
            'EVAL_BATCH_SIZE': 'evaluation.batch_size',
            'EVAL_RANDOM_SEED': 'reproducibility.random_seed',
            'EVAL_OUTPUT_DIR': 'output_dir',
            'EVAL_PROJECT_PATH': 'project_path',
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(config_path, env_value)
    
    def _set_nested_value(self, path: str, value: Any):
        """Set a nested configuration value using dot notation."""
        keys = path.split('.')
        current = self._config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Type conversion for numeric values
        if keys[-1] in ['max_concurrent_tasks', 'batch_size', 'random_seed']:
            try:
                value = int(value)
            except ValueError:
                pass
        
        current[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        current = self._config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        self._set_nested_value(key, value)
    
    def get_model_config(self, model_type: str) -> ModelConfig:
        """Get model configuration."""
        model_data = self.get(f"{model_type}_model", {})
        return ModelConfig(**model_data)
    
    def get_evaluation_settings(self) -> EvaluationSettings:
        """Get evaluation settings."""
        eval_data = self.get("evaluation", {})
        return EvaluationSettings(**eval_data)
    
    def get_reproducibility_config(self) -> ReproducibilityConfig:
        """Get reproducibility configuration."""
        repro_data = self.get("reproducibility", {})
        return ReproducibilityConfig(**repro_data)
    
    def get_resource_config(self) -> ResourceConfig:
        """Get resource configuration."""
        resource_data = self.get("resources", {})
        return ResourceConfig(**resource_data)
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        logging_data = self.get("logging", {})
        return LoggingConfig(**logging_data)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        try:
            jsonschema.validate(self._config, self.CONFIG_SCHEMA)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        # Additional business logic validation
        eval_settings = self.get_evaluation_settings()
        if not eval_settings.methods:
            errors.append("At least one evaluation method must be specified")
        
        # Check required model configurations
        llm_config = self.get_model_config("llm")
        if not llm_config.name:
            errors.append("LLM model name is required")
        
        embedding_config = self.get_model_config("embedding")
        if not embedding_config.name:
            errors.append("Embedding model name is required")
        
        # Check paths
        project_path = self.get("project_path")
        if project_path and not os.path.exists(project_path):
            errors.append(f"Project path does not exist: {project_path}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return self._config.copy()
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        
        try:
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                elif config_path.suffix.lower() == '.json':
                    json.dump(self._config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            self.logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_path}: {str(e)}")
            raise


def create_default_config(environment: Environment = Environment.DEVELOPMENT) -> UnifiedConfiguration:
    """Create a default configuration."""
    config = UnifiedConfiguration(environment)
    
    # Set default values
    default_config = {
        "environment": environment.value,
        "llm_model": {
            "name": "gpt-4-turbo",
            "type": "openai",
            "api_base_url": "https://api.openai.com/v1",
            "max_tokens": 4096,
            "temperature": 0.1,
            "timeout": 30
        },
        "embedding_model": {
            "name": "BAAI/bge-large-en-v1.5",
            "type": "huggingface",
            "timeout": 30
        },
        "judge_llm_model": {
            "name": "llama4_chat",
            "type": "mistral",
            "timeout": 30
        },
        "evaluation": {
            "methods": ["local_search", "global_search", "basic_search"],
            "max_concurrent_tasks": 10,
            "batch_size": 5,
            "enable_caching": True,
            "cache_ttl": 3600
        },
        "reproducibility": {
            "random_seed": 42,
            "evaluation_order_seed": 123,
            "enable_deterministic_ordering": True,
            "track_versions": True
        },
        "resources": {
            "cleanup_temp_files": True,
            "connection_pool_size": 10,
            "request_timeout": 30
        },
        "logging": {
            "level": "INFO",
            "enable_structured_logging": True,
            "enable_metrics": True
        }
    }
    
    config._config = default_config
    return config
