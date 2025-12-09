"""
Reproducibility controls for deterministic evaluation.
"""

import random
import numpy as np
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import datetime


@dataclass
class ReproducibilityState:
    """State information for reproducibility tracking."""
    random_seed: int
    evaluation_order_seed: int
    python_version: str
    framework_version: str
    model_versions: Dict[str, str]
    dataset_hash: str
    config_hash: str
    timestamp: str
    environment: str


class ReproducibilityManager:
    """Manages reproducibility controls for evaluation."""
    
    def __init__(self, random_seed: int = 42, evaluation_order_seed: int = 123,
                 enable_deterministic_ordering: bool = True):
        self.random_seed = random_seed
        self.evaluation_order_seed = evaluation_order_seed
        self.enable_deterministic_ordering = enable_deterministic_ordering
        self.logger = logging.getLogger(__name__)
        self._state: Optional[ReproducibilityState] = None
    
    def initialize_seeds(self) -> None:
        """Initialize all random seeds for reproducible execution."""
        # Set Python's built-in random seed
        random.seed(self.random_seed)
        
        # Set NumPy random seed
        np.random.seed(self.random_seed)
        
        # Set hash seed for string hashing (note: this needs to be set before Python starts)
        # os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        
        self.logger.info(f"Initialized random seeds: {self.random_seed}")
    
    def create_deterministic_order(self, items: List[Any], 
                                 key_func: Optional[Callable[[Any], str]] = None) -> List[Any]:
        """Create a deterministic ordering of items."""
        if not self.enable_deterministic_ordering:
            return items
        
        # Create a local random generator with evaluation order seed
        rng = random.Random(self.evaluation_order_seed)
        
        # If no key function provided, use string representation
        if key_func is None:
            key_func = lambda x: str(x)
        
        # Create list of (deterministic_key, random_tie_breaker, item) tuples
        keyed_items = [
            (key_func(item), rng.random(), item) 
            for item in items
        ]
        
        # Sort by deterministic key first, then by random tie-breaker
        keyed_items.sort(key=lambda x: (x[0], x[1]))
        
        # Return just the items
        return [item for _, _, item in keyed_items]
    
    def hash_dataset(self, dataset: List[Dict[str, Any]]) -> str:
        """Create a hash of the dataset for reproducibility tracking."""
        # Sort dataset by id for consistent hashing
        sorted_dataset = sorted(dataset, key=lambda x: x.get('id', ''))
        
        # Create a simplified representation for hashing
        hash_data = []
        for item in sorted_dataset:
            hash_item = {
                'id': item.get('id', ''),
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'question_type': item.get('question_type', '')
            }
            hash_data.append(hash_item)
        
        # Convert to JSON string and hash
        json_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def hash_config(self, config: Dict[str, Any]) -> str:
        """Create a hash of the configuration for reproducibility tracking."""
        # Remove dynamic fields that shouldn't affect reproducibility
        config_copy = config.copy()
        dynamic_keys = ['timestamp', 'output_dir', 'logs_file']
        for key in dynamic_keys:
            config_copy.pop(key, None)
        
        json_str = json.dumps(config_copy, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def capture_state(self, dataset: List[Dict[str, Any]], 
                     config: Dict[str, Any], environment: str) -> ReproducibilityState:
        """Capture current state for reproducibility."""
        import sys
        
        # Try to get actual framework version
        framework_version = self._get_framework_version()
        
        # Try to capture model versions
        model_versions = self._capture_model_versions()
        
        self._state = ReproducibilityState(
            random_seed=self.random_seed,
            evaluation_order_seed=self.evaluation_order_seed,
            python_version=sys.version,
            framework_version=framework_version,
            model_versions=model_versions,
            dataset_hash=self.hash_dataset(dataset),
            config_hash=self.hash_config(config),
            timestamp=datetime.datetime.now().isoformat(),
            environment=environment
        )
        
        return self._state
    
    def _get_framework_version(self) -> str:
        """Get the actual framework version."""
        try:
            # Try to get version from package metadata
            import importlib.metadata
            try:
                return importlib.metadata.version('aime-graphrag-evaluate')
            except importlib.metadata.PackageNotFoundError:
                pass
            
            # Try to get version from git
            import subprocess
            result = subprocess.run(['git', 'describe', '--tags', '--always'], 
                                  capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            if result.returncode == 0:
                return f"git-{result.stdout.strip()}"
            
            # Fallback to reading version file
            version_file = Path(__file__).parent.parent / 'VERSION'
            if version_file.exists():
                return version_file.read_text().strip()
                
        except Exception as e:
            self.logger.debug(f"Could not determine framework version: {e}")
        
        return "1.0.0-unknown"
    
    def _capture_model_versions(self) -> Dict[str, str]:
        """Capture model versions from configuration."""
        try:
            model_versions = {}
            
            # Try to get OpenAI model info
            try:
                import openai
                model_versions['openai_client_version'] = openai.__version__
            except ImportError:
                pass
            
            # Try to get transformers info
            try:
                import transformers
                model_versions['transformers_version'] = transformers.__version__
            except ImportError:
                pass
            
            # Try to get langchain info
            try:
                import langchain
                model_versions['langchain_version'] = langchain.__version__
            except ImportError:
                pass
            
            return model_versions
            
        except Exception as e:
            self.logger.debug(f"Could not capture model versions: {e}")
            return {}
    
    def save_state(self, output_dir: str) -> None:
        """Save reproducibility state to file."""
        if self._state is None:
            self.logger.warning("No state captured, cannot save")
            return
        
        state_file = Path(output_dir) / "reproducibility_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(state_file, 'w') as f:
                json.dump(self._state.__dict__, f, indent=2)
            
            self.logger.info(f"Saved reproducibility state to {state_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save reproducibility state: {str(e)}")
    
    def load_state(self, state_file: str) -> Optional[ReproducibilityState]:
        """Load reproducibility state from file."""
        try:
            with open(state_file, 'r') as f:
                state_dict = json.load(f)
            
            self._state = ReproducibilityState(**state_dict)
            self.logger.info(f"Loaded reproducibility state from {state_file}")
            return self._state
            
        except Exception as e:
            self.logger.error(f"Failed to load reproducibility state: {str(e)}")
            return None
    
    def validate_reproducibility(self, dataset: List[Dict[str, Any]], 
                               config: Dict[str, Any]) -> List[str]:
        """Validate that current setup matches saved state."""
        if self._state is None:
            return ["No saved state available for validation"]
        
        errors = []
        
        # Check dataset hash
        current_dataset_hash = self.hash_dataset(dataset)
        if current_dataset_hash != self._state.dataset_hash:
            errors.append(f"Dataset hash mismatch: {current_dataset_hash} != {self._state.dataset_hash}")
        
        # Check config hash
        current_config_hash = self.hash_config(config)
        if current_config_hash != self._state.config_hash:
            errors.append(f"Config hash mismatch: {current_config_hash} != {self._state.config_hash}")
        
        # Check random seed
        if self.random_seed != self._state.random_seed:
            errors.append(f"Random seed mismatch: {self.random_seed} != {self._state.random_seed}")
        
        return errors
    
    def get_state(self) -> Optional[ReproducibilityState]:
        """Get current reproducibility state."""
        return self._state


class VersionTracker:
    """Tracks versions of models, datasets, and frameworks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._versions: Dict[str, str] = {}
    
    def track_model_version(self, model_name: str, version: str) -> None:
        """Track a model version."""
        self._versions[f"model_{model_name}"] = version
        self.logger.info(f"Tracking model version: {model_name} = {version}")
    
    def track_dataset_version(self, dataset_name: str, version: str) -> None:
        """Track a dataset version."""
        self._versions[f"dataset_{dataset_name}"] = version
        self.logger.info(f"Tracking dataset version: {dataset_name} = {version}")
    
    def track_framework_version(self, framework_name: str, version: str) -> None:
        """Track a framework version."""
        self._versions[f"framework_{framework_name}"] = version
        self.logger.info(f"Tracking framework version: {framework_name} = {version}")
    
    def get_versions(self) -> Dict[str, str]:
        """Get all tracked versions."""
        return self._versions.copy()
    
    def save_versions(self, output_dir: str) -> None:
        """Save version information to file."""
        versions_file = Path(output_dir) / "versions.json"
        versions_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(versions_file, 'w') as f:
                json.dump(self._versions, f, indent=2)
            
            self.logger.info(f"Saved version information to {versions_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save version information: {str(e)}")


def create_reproducibility_manager(config: Dict[str, Any]) -> ReproducibilityManager:
    """Create a reproducibility manager from configuration."""
    repro_config = config.get('reproducibility', {})
    
    return ReproducibilityManager(
        random_seed=repro_config.get('random_seed', 42),
        evaluation_order_seed=repro_config.get('evaluation_order_seed', 123),
        enable_deterministic_ordering=repro_config.get('enable_deterministic_ordering', True)
    )
