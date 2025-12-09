import os
import yaml
import logging
from typing import Any, Dict, Optional
from pathlib import Path

# Import GraphRAG components
try:
    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag.language_model.manager import ModelManager as GraphRAGModelManager
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    logging.warning("GraphRAG not available, some features may be limited")

# Import AIME API components
try:
    from aime_api_client_interface.model_api import ModelAPI
    AIME_AVAILABLE = True
except ImportError:
    AIME_AVAILABLE = False
    logging.warning("AIME API not available, some features may be limited")


class ModelManager:
    """Centralized model management for evaluation framework."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.graphrag_model_manager = None
        self.config = None
        
        if GRAPHRAG_AVAILABLE:
            self.graphrag_model_manager = GraphRAGModelManager()
    
    def load_config(self, config_path: str, project_path: Optional[str] = None) -> Any:
        """Load GraphRAG configuration."""
        if not GRAPHRAG_AVAILABLE:
            raise ImportError("GraphRAG is not available")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Update database URI if project path is provided
            if project_path:
                lancedb_path = os.path.join(project_path, 'output', 'lancedb')
                if 'vector_store' in config_dict and 'default_vector_store' in config_dict['vector_store']:
                    config_dict['vector_store']['default_vector_store']['db_uri'] = lancedb_path
                elif 'default_vector_store' in config_dict:
                    config_dict['default_vector_store']['db_uri'] = lancedb_path
            
            self.config = GraphRagConfig(**config_dict)
            self.logger.info(f"Loaded GraphRAG configuration from {config_path}")
            return self.config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
            raise
    
    def get_chat_model(self, model_id: str = None) -> Any:
        """Get chat model instance."""
        if not GRAPHRAG_AVAILABLE or not self.config:
            raise ImportError("GraphRAG is not available or config not loaded")
        
        try:
            if model_id is None:
                # Try to get default chat model from config
                if hasattr(self.config, 'local_search') and hasattr(self.config.local_search, 'chat_model_id'):
                    model_id = self.config.local_search.chat_model_id
                else:
                    raise ValueError("No chat model ID specified and no default found in config")
            
            model_config = self.config.get_language_model_config(model_id)
            llm = self.graphrag_model_manager.get_or_create_chat_model(
                name="evaluation_llm",
                model_type=model_config.type,
                config=model_config,
            )
            
            self.logger.info(f"Created chat model: {model_id}")
            return llm
            
        except Exception as e:
            self.logger.error(f"Failed to create chat model {model_id}: {str(e)}")
            raise
    
    def get_embedding_model(self, model_id: str = None) -> Any:
        """Get embedding model instance."""
        if not GRAPHRAG_AVAILABLE or not self.config:
            raise ImportError("GraphRAG is not available or config not loaded")
        
        try:
            if model_id is None:
                # Try to get default embedding model from config
                if hasattr(self.config, 'local_search') and hasattr(self.config.local_search, 'embedding_model_id'):
                    model_id = self.config.local_search.embedding_model_id
                else:
                    raise ValueError("No embedding model ID specified and no default found in config")
            
            model_config = self.config.get_language_model_config(model_id)
            embeddings = self.graphrag_model_manager.get_or_create_embedding_model(
                name="evaluation_embeddings",
                model_type=model_config.type,
                config=model_config,
            )
            
            self.logger.info(f"Created embedding model: {model_id}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to create embedding model {model_id}: {str(e)}")
            raise
    
    def get_judge_chat_model(self, judge_model_name: str = "llama4_chat", 
                           api_base_url: Optional[str] = None,
                           api_key: Optional[str] = None,
                           project_path: Optional[str] = None) -> Any:
        """Get chat model instance specifically for LLM-as-a-judge functionality."""
        try:
            # For AIME models like mistral_chat, llama4_chat, or gpt_oss_chat, try to extract config from yaml file first
            if (judge_model_name in ["mistral_chat", "llama4_chat", "gpt_oss_chat"] or 
                "mistral" in judge_model_name.lower() or "llama" in judge_model_name.lower() or 
                "gpt" in judge_model_name.lower()):
                aime_config = None
                
                # Try to extract from project config files if project path is provided
                if project_path:
                    config_path = os.path.join(project_path, 'settings.yaml')
                    if os.path.exists(config_path):
                        try:
                            aime_config = self.extract_aime_config(config_path)
                            aime_config['endpoint'] = judge_model_name  # Override endpoint
                        except Exception as e:
                            self.logger.warning(f"Could not extract AIME config from {config_path}: {str(e)}")
                
                # If we couldn't extract from config file, try using the GraphRAG config
                if not aime_config and self.config:
                    try:
                        # Get the default chat model config to extract AIME settings
                        default_model_config = self.config.get_language_model_config('default_chat_model')
                        
                        # Extract values and ensure they're not empty strings
                        api_server_val = getattr(default_model_config, 'api_url', 
                                                getattr(default_model_config, 'api_server', 'https://api.aime.info'))
                        user_val = getattr(default_model_config, 'email', 
                                          getattr(default_model_config, 'user', ''))
                        key_val = getattr(default_model_config, 'api_key', 
                                         getattr(default_model_config, 'key', ''))
                        
                        # Only create config if we have non-empty values
                        if api_server_val and user_val and key_val:
                            aime_config = {
                                'api_server': api_server_val.rstrip('/') if isinstance(api_server_val, str) else api_server_val,
                                'endpoint': judge_model_name,  # Use judge model name
                                'user': user_val,
                                'api_key': key_val
                            }
                        else:
                            self.logger.warning(f"GraphRAG config has incomplete AIME settings (api_server: {bool(api_server_val)}, user: {bool(user_val)}, key: {bool(key_val)})")
                    except Exception as e:
                        self.logger.warning(f"Could not extract AIME config from GraphRAG config: {str(e)}")
                
                # Override with provided parameters if available
                if api_base_url and aime_config:
                    aime_config['api_server'] = api_base_url.rstrip('/')
                elif api_base_url:
                    # Only create config if we have all required values
                    env_key = os.environ.get('AIME_API_KEY', '')
                    final_key = api_key or env_key
                    if final_key:  # Only create if we have a key
                        aime_config = {
                            'api_server': api_base_url.rstrip('/'),
                            'endpoint': judge_model_name,
                            'user': 'evaluation',
                            'api_key': final_key
                        }
                    else:
                        self.logger.warning("Cannot create AIME config: no API key provided via api_key parameter or AIME_API_KEY environment variable")
                
                if api_key and aime_config:
                    aime_config['api_key'] = api_key
                
                # Try to create AIME model if available and config is valid
                if AIME_AVAILABLE and aime_config:
                    # Log sanitized config (hide sensitive info)
                    sanitized_config = {
                        'api_server': aime_config.get('api_server'),
                        'endpoint': aime_config.get('endpoint'),
                        'user': aime_config.get('user'),
                        'api_key': '***' if aime_config.get('api_key') else None
                    }
                    self.logger.info(f"AIME config for judge model: {sanitized_config}")
                    
                    # Check which values are missing or empty
                    missing_values = [k for k, v in aime_config.items() if not v or (isinstance(v, str) and not v.strip())]
                    if missing_values:
                        self.logger.warning(f"AIME config missing or empty values for: {missing_values}")
                    
                    # Check that all values are present and not empty strings
                    if all(v and (not isinstance(v, str) or v.strip()) for v in aime_config.values()):
                        try:
                            model_api = self.create_aime_api_client(**aime_config)
                            self.logger.info(f"Created AIME judge model: {judge_model_name} at {aime_config['api_server']}")
                            return model_api
                        except Exception as e:
                            self.logger.error(f"Failed to create AIME judge model: {str(e)}")
                            import traceback
                            self.logger.error(f"Traceback: {traceback.format_exc()}")
                            self.logger.warning(f"Falling back to GraphRAG model manager")
                    else:
                        self.logger.warning(f"Incomplete AIME configuration - using GraphRAG fallback")
                else:
                    if not AIME_AVAILABLE:
                        self.logger.warning("AIME API not available - using GraphRAG fallback")
                    if not aime_config:
                        self.logger.warning("Could not extract AIME configuration - using GraphRAG fallback")
            
            # Fallback to GraphRAG model manager if available
            if GRAPHRAG_AVAILABLE and self.config and self.graphrag_model_manager:
                try:
                    # Try to find a suitable model in the config, or use default
                    model_id = judge_model_name
                    if hasattr(self.config, 'local_search') and hasattr(self.config.local_search, 'chat_model_id'):
                        model_id = self.config.local_search.chat_model_id
                    
                    model_config = self.config.get_language_model_config(model_id)
                    
                    # Override with judge-specific settings
                    if api_base_url:
                        model_config.api_base = api_base_url
                    if api_key:
                        model_config.api_key = api_key
                    
                    # Use lower temperature for more consistent judging
                    model_config.temperature = 0.1
                    model_config.max_tokens = 1024
                    
                    llm = self.graphrag_model_manager.get_or_create_chat_model(
                        name="judge_llm",
                        model_type=model_config.type,
                        config=model_config,
                    )
                    
                    self.logger.info(f"Created GraphRAG judge model: {model_id}")
                    return llm
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create GraphRAG judge model: {str(e)}")
            
            # If all else fails, create a simple wrapper
            self.logger.warning(f"Using fallback judge model configuration for {judge_model_name}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to create judge model {judge_model_name}: {str(e)}")
            raise
    
    def create_aime_api_client(self, api_server: str, endpoint: str, 
                              user: str, api_key: str) -> Any:
        """Create AIME API client.
        
        Args:
            api_server: The AIME API server URL (e.g., https://api.aime.info)
            endpoint: The endpoint/model name (e.g., llama4_chat, mistral_chat)
            user: The user email for authentication
            api_key: The API key for authentication
            
        Returns:
            ModelAPI: Configured AIME API client instance with _aime_user and _aime_api_key attributes
        """
        if not AIME_AVAILABLE:
            raise ImportError("AIME API is not available - ensure aime_api_client_interface is installed")
        
        try:
            self.logger.info("=" * 80)
            self.logger.info("CREATING AIME API CLIENT")
            self.logger.info("=" * 80)
            
            # Validate inputs - ensure they are not None and not empty strings
            if not api_server or not api_server.strip():
                raise ValueError("api_server is required and cannot be empty")
            if not endpoint or not endpoint.strip():
                raise ValueError("endpoint is required and cannot be empty")
            if not user or not user.strip():
                raise ValueError("user is required and cannot be empty")
            if not api_key or not api_key.strip():
                raise ValueError("api_key is required and cannot be empty")
            
            # Log configuration (sanitized)
            self.logger.info(f"API Server: {api_server}")
            self.logger.info(f"Endpoint: {endpoint}")
            self.logger.info(f"User: {user}")
            self.logger.info(f"API Key: {'*' * min(len(api_key), 10)}... (length: {len(api_key)})")
            self.logger.info("-" * 80)
            
            # Log the exact parameters being passed to ModelAPI
            self.logger.info("ModelAPI initialization parameters:")
            self.logger.info(f"  api_server={api_server.rstrip('/')}")
            self.logger.info(f"  endpoint_name={endpoint}")
            self.logger.info(f"  user={user}")
            self.logger.info(f"  api_key=[REDACTED]")
            
            # Create the ModelAPI instance - use 'api_key' parameter
            model_api = ModelAPI(
                api_server=api_server.rstrip('/'),
                endpoint_name=endpoint,
                user=user,
                api_key=api_key
            )
            
            # IMPORTANT: Store credentials on the object since ModelAPI doesn't store 'user'
            # These will be used by the AIMEAPIAdapter for login
            model_api._aime_user = user
            model_api._aime_api_key = api_key
            
            # Log the created object attributes
            self.logger.info("-" * 80)
            self.logger.info("ModelAPI instance created with attributes:")
            self.logger.info(f"  api_server: {getattr(model_api, 'api_server', 'N/A')}")
            self.logger.info(f"  endpoint_name: {getattr(model_api, 'endpoint_name', 'N/A')}")
            self.logger.info(f"  user (stored as _aime_user): {getattr(model_api, '_aime_user', 'N/A')}")
            self.logger.info(f"  api_key: {'***' if getattr(model_api, 'api_key', getattr(model_api, '_aime_api_key', None)) else 'N/A'}")
            self.logger.info(f"  Has session: {bool(getattr(model_api, 'session', None))}")
            
            self.logger.info("✓ AIME API client created successfully")
            self.logger.info("=" * 80)
            return model_api
            
        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error("AIME API CLIENT CREATION FAILED")
            self.logger.error("=" * 80)
            self.logger.error(f"Exception Type: {type(e).__name__}")
            self.logger.error(f"Exception Message: {str(e)}")
            import traceback
            self.logger.error("Traceback:")
            self.logger.error(traceback.format_exc())
            self.logger.error("=" * 80)
            raise
    
    def extract_aime_config(self, config_path: str) -> Dict[str, str]:
        """Extract AIME API configuration from config file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Try to extract AIME configuration
            if 'models' in config_dict and 'default_chat_model' in config_dict['models']:
                chat_model_cfg = config_dict['models']['default_chat_model']
                
                api_server = (
                    chat_model_cfg.get('api_url') or 
                    chat_model_cfg.get('api_server')
                )
                model_name = chat_model_cfg.get('model')
                user = chat_model_cfg.get('email') or chat_model_cfg.get('user')
                key = chat_model_cfg.get('api_key') or chat_model_cfg.get('key')
                
                # Filter out empty strings and check all values are present
                api_server = api_server.strip() if isinstance(api_server, str) and api_server else None
                model_name = model_name.strip() if isinstance(model_name, str) and model_name else None
                user = user.strip() if isinstance(user, str) and user else None
                key = key.strip() if isinstance(key, str) and key else None
                
                if all([api_server, model_name, user, key]):
                    return {
                        'api_server': api_server,
                        'endpoint': model_name,
                        'user': user,
                        'api_key': key
                    }
                else:
                    missing = []
                    if not api_server: missing.append('api_server/api_url')
                    if not model_name: missing.append('model')
                    if not user: missing.append('user/email')
                    if not key: missing.append('key/api_key')
                    raise ValueError(f"Incomplete AIME configuration - missing: {', '.join(missing)}")
            
            raise ValueError("AIME configuration not found in config file (no models.default_chat_model section)")
            
        except Exception as e:
            self.logger.error(f"Failed to extract AIME configuration: {str(e)}")
            raise
    
    def load_index_files(self, output_dir: str) -> Dict[str, Any]:
        """Load GraphRAG index files.
        
        Also filters out corrupted/error community reports that may degrade search quality.
        """
        if not GRAPHRAG_AVAILABLE:
            raise ImportError("GraphRAG is not available")
        
        try:
            import pandas as pd
            
            files = {}
            file_names = [
                'entities.parquet',
                'communities.parquet', 
                'community_reports.parquet',
                'text_units.parquet',
                'relationships.parquet',
                'covariates.parquet'
            ]
            
            for fname in file_names:
                fpath = os.path.join(output_dir, fname)
                if os.path.exists(fpath):
                    df = pd.read_parquet(fpath)
                    
                    # Filter out error reports from community_reports
                    if fname == 'community_reports.parquet' and df is not None:
                        initial_count = len(df)
                        
                        # Combine all error detection patterns to avoid double-counting
                        # Most errors have both error title AND error content
                        error_mask = pd.Series(False, index=df.index)
                        
                        # Check for error patterns in title
                        if 'title' in df.columns:
                            error_mask |= df['title'].str.contains('Error Generating Report', na=False, case=False)
                        
                        # Check for error patterns in content (use correct column name)
                        content_col = 'full_content' if 'full_content' in df.columns else 'content'
                        if content_col in df.columns:
                            error_mask |= df[content_col].str.contains('An error occurred:', na=False, case=False)
                            error_mask |= df[content_col].str.contains('Report generation failed', na=False, case=False)
                            error_mask |= df[content_col].str.contains('Lost connection while receiving progress', na=False, case=False)
                            # Short error messages (less than 200 chars with "Error:")
                            error_mask |= (df[content_col].str.contains('Error:', na=False, case=False) & 
                                          (df[content_col].str.len() < 200))
                        
                        # Apply the combined filter
                        df = df[~error_mask]
                        
                        filtered_count = len(df)
                        error_count = initial_count - filtered_count
                        if error_count > 0:
                            error_rate = (error_count / initial_count) * 100
                            log_level = 'error' if error_rate > 10 else 'warning'
                            msg = (
                                f"Filtered {error_count} error reports from community_reports "
                                f"({filtered_count} remaining out of {initial_count}) - {error_rate:.1f}% corrupted"
                            )
                            if log_level == 'error':
                                self.logger.error(msg + " - HIGH ERROR RATE indicates indexing issues!")
                            else:
                                self.logger.warning(msg)
                    
                    files[fname.split('.')[0]] = df
                    self.logger.info(f"Loaded index file: {fname}")
                else:
                    files[fname.split('.')[0]] = None
                    self.logger.warning(f"Index file not found: {fname}")
            
            return files
            
        except Exception as e:
            self.logger.error(f"Failed to load index files from {output_dir}: {str(e)}")
            raise
    
    def validate_environment(self) -> Dict[str, bool]:
        """Validate that all required components are available."""
        validation = {
            'graphrag_available': GRAPHRAG_AVAILABLE,
            'aime_available': AIME_AVAILABLE,
            'config_loaded': self.config is not None
        }
        
        self.logger.info(f"Environment validation: {validation}")
        return validation 