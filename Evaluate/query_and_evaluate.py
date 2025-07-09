import argparse
import json
import os
import asyncio
import pandas as pd
from pathlib import Path
import yaml
import tqdm
import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import re

# Constants
ENTITIES_FILE = 'entities.parquet'
COMMUNITIES_FILE = 'communities.parquet'
COMMUNITY_REPORTS_FILE = 'community_reports.parquet'
TEXT_UNITS_FILE = 'text_units.parquet'
RELATIONSHIPS_FILE = 'relationships.parquet'
COVARIATES_FILE = 'covariates.parquet'
LOGS_FILENAME = 'logs.json'
DEFAULT_METHODS = ['local_search', 'global_search']
MAX_CONTEXT_LEN = 300

BATCH_SIZE = 10  # Number of questions after which to write results/logs

# Set up logging
logger = logging.getLogger("query_and_evaluate")
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

# File handler will be added in main() after output dir is known

from graphrag.api.query import local_search, global_search, basic_search
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.language_model.manager import ModelManager
from aime_api_client_interface.model_api import ModelAPI

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'Datasets/GraphRAG-Benchmark/Evaluation'))
from metrics_utils import (
    exact_match,
    f1_score,
    hallucination_rate,
    extract_triples_llm,
    compute_triple_metrics,
    compute_all_metrics,
)

def load_index_files(output_dir: str) -> Dict[str, Optional[pd.DataFrame]]:
    """Load all required index files as DataFrames from the output directory."""
    files = {}
    for fname in [
        ENTITIES_FILE,
        COMMUNITIES_FILE,
        COMMUNITY_REPORTS_FILE,
        TEXT_UNITS_FILE,
        RELATIONSHIPS_FILE,
        COVARIATES_FILE,
    ]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            files[fname.split('.')[0]] = pd.read_parquet(fpath)
        else:
            files[fname.split('.')[0]] = None
    return files

def load_config(config_path: str, project_path: Optional[str] = None) -> 'GraphRagConfig':
    """Load GraphRAG config from YAML and update db_uri to point to the LanceDB in the output folder at project_path."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    lancedb_path = os.path.join(project_path, 'output', 'lancedb') if project_path else None
    if project_path is not None:
        if 'vector_store' in config_dict and 'default_vector_store' in config_dict['vector_store']:
            config_dict['vector_store']['default_vector_store']['db_uri'] = lancedb_path
        elif 'default_vector_store' in config_dict:
            config_dict['default_vector_store']['db_uri'] = lancedb_path
    return GraphRagConfig(**config_dict)

def truncate_context(context: Union[str, Any], maxlen: int = MAX_CONTEXT_LEN) -> str:
    """Truncate context string for results output."""
    c = str(context)
    return c[:maxlen] + ("..." if len(c) > maxlen else "")

def log_error(logs: List[dict], message: str, question: str, method: str, level: str = 'error') -> None:
    """Log an error or warning to the logs list and logger."""
    entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'type': level,
        'message': message,
        'question': question,
        'method': method
    }
    logs.append(entry)
    if level == 'error':
        logger.error(f"{method} | {question} | {message}")
    elif level == 'warning':
        logger.warning(f"{method} | {question} | {message}")
    else:
        logger.info(f"{method} | {question} | {message}")

class LLMInvokeAdapter:
    def __init__(self, llm):
        self.llm = llm
    async def ainvoke(self, prompt, config=None):
        response = await self.llm.achat(prompt)
        if not hasattr(response, 'content'):
            if hasattr(response, 'output') and hasattr(response.output, 'content'):
                response.content = response.output.content
            elif hasattr(response, '__str__'):
                response.content = str(response)
        return response

async def run_method_on_question(
    method: str,
    q: str,
    gt: str,
    item: dict,
    config: Any,
    index_files: Dict[str, Optional[pd.DataFrame]],
    novel_contexts: Dict[str, str],
    model_manager: Any,
    llm_adapter: Any,
    embeddings_adapter: Any,
    config_path: str,
    logs: List[dict]
) -> Tuple[str, List[str], str]:
    """Run a single method on a question and return (answer, context_strs, error_message)."""
    answer = None
    context = None
    error_message = ''
    source = item.get('source')
    try:
        if method == 'local_search':
            answer, context = await local_search(
                config=config,
                entities=index_files['entities'],
                communities=index_files['communities'],
                community_reports=index_files['community_reports'],
                text_units=index_files['text_units'],
                relationships=index_files['relationships'],
                covariates=index_files['covariates'],
                community_level=2,
                response_type='One Word',
                query=q,
            )
        elif method == 'global_search':
            answer, context = await global_search(
                config=config,
                entities=index_files['entities'],
                communities=index_files['communities'],
                community_reports=index_files['community_reports'],
                community_level=2,
                dynamic_community_selection=False,
                response_type='One Word',
                query=q,
            )
        elif method == 'basic_search':
            answer, context = await basic_search(
                config=config,
                text_units=index_files['text_units'],
                query=q,
            )
        elif method == 'llm_with_context':
            context = novel_contexts.get(source, None)
            if not context:
                answer = f"[ERROR: No context loaded from novel.json for source: {source}]"
                log_error(logs, f'No context loaded from novel.json for source: {source}', q, method)
            else:
                prompt = f"[CONTEXT]: {context}\n[QUESTION]: {q}"
                try:
                    chat_model_cfg = None
                    api_server = model_name = user = key = None
                    # Try dataclass first
                    if hasattr(config, 'models') and hasattr(config.models, 'default_chat_model'):
                        chat_model_cfg = getattr(config.models, 'default_chat_model', None)
                        if chat_model_cfg:
                            api_server = getattr(chat_model_cfg, 'api_url', None) or getattr(chat_model_cfg, 'api_server', None)
                            model_name = getattr(chat_model_cfg, 'model', None)
                            user = getattr(chat_model_cfg, 'email', None) or getattr(chat_model_cfg, 'user', None)
                            key = getattr(chat_model_cfg, 'api_key', None) or getattr(chat_model_cfg, 'key', None)
                    # Fallback: use config_dict from load_config
                    if not (api_server and model_name and user and key):
                        # Try to get config_dict from closure
                        config_dict = None
                        try:
                            import inspect
                            outer = inspect.currentframe().f_back
                            config_dict = outer.f_locals.get('config_dict', None)
                        except Exception:
                            pass
                        if not config_dict:
                            # Try to reload YAML
                            with open(config_path, 'r') as f:
                                config_dict = yaml.safe_load(f)
                        if config_dict and 'models' in config_dict and 'default_chat_model' in config_dict['models']:
                            chat_model_cfg = config_dict['models']['default_chat_model']
                            api_server = chat_model_cfg.get('api_url', None) or chat_model_cfg.get('api_server', None)
                            model_name = chat_model_cfg.get('model', None)
                            user = chat_model_cfg.get('email', None) or chat_model_cfg.get('user', None)
                            key = chat_model_cfg.get('api_key', None) or chat_model_cfg.get('key', None)
                    # DEBUG: Log extracted values
                    logger.debug(f'Extracted api_server={api_server}, model_name={model_name}, user={user}, key={key}')
                    if not (api_server and model_name and user and key):
                        logger.debug(f'Full chat_model_cfg: {repr(chat_model_cfg)}')
                        answer = '[ERROR: Missing AIME API credentials or model info in config]'
                        log_error(logs, f'Missing AIME API credentials or model info in config', q, method)
                    else:
                        endpoint = model_name
                        model_api = ModelAPI(api_server=api_server.rstrip('/'), endpoint_name=endpoint, user=user, key=key)
                        await model_api.do_api_login_async()
                        params = {"prompt_input": prompt}
                        output_generator = model_api.get_api_request_generator(params)
                        answer = None
                        progress_error = {'occurred': False, 'message': ''}
                        def progress_error_callback(error_description):
                            progress_error['occurred'] = True
                            progress_error['message'] = error_description
                            log_error(logs, f'Progress error: {error_description}', q, method)
                        try:
                            async for progress in output_generator:
                                if progress_error['occurred']:
                                    break
                                logger.debug(f'AIME API progress: {repr(progress)}')
                                if isinstance(progress, dict) and progress.get('job_state') == 'done':
                                    result_data = progress.get('result_data', {})
                                    answer = result_data.get('text', None) or result_data.get('output', None) or str(result_data)
                        except Exception as e:
                            progress_error_callback(str(e))
                        if progress_error['occurred']:
                            answer = f"[ERROR: LLM call failed: {progress_error['message']}]"
                        if not answer or answer.strip() == '' or answer.startswith('[ERROR'):
                            log_error(logs, f'No response text received from AIME API (stream).', q, method)
                            answer = '[ERROR: LLM call failed: No response text received from AIME API]'
                except Exception as e:
                    answer = f"[ERROR: LLM call failed: {str(e)}]"
                    log_error(logs, f'LLM call failed: {str(e)}', q, method)
        else:
            error_message = f"Unknown method: {method}"
            log_error(logs, error_message, q, method)
    except Exception as e:
        error_message = str(e)
        log_error(logs, f'Exception in method {method}: {error_message}', q, method)
    # Convert context to list of strings
    context_strs = []
    if isinstance(context, dict):
        for v in context.values():
            if isinstance(v, list):
                context_strs.extend([str(x) for x in v])
            else:
                context_strs.append(str(v))
    elif isinstance(context, list):
        context_strs = [str(x) for x in context]
    elif context is not None:
        context_strs = [str(context)]
    return answer, context_strs, error_message

async def query_and_evaluate(
    config_path: str,
    output_dir: str,
    questions_path: str,
    results_path: str,
    method_list: List[str] = DEFAULT_METHODS,
    project_path: Optional[str] = None,
    input_json_path: Optional[str] = None
) -> None:
    """
    Run all methods on all questions, compute metrics, and save results and logs.
    Args:
        config_path: Path to the config YAML file.
        output_dir: Path to the output directory.
        questions_path: Path to the questions JSON file.
        results_path: Path to save results JSON.
        method_list: List of methods to run.
        project_path: Project root path.
        input_json_path: Path to novel.json for llm_with_context baseline.
    """
    # Load config and index files
    config = load_config(config_path, project_path=project_path)
    index_files = load_index_files(output_dir)

    # Instantiate LLM for metrics and for LLM baseline
    model_manager = ModelManager()
    model_config = config.get_language_model_config(config.local_search.chat_model_id)
    llm = model_manager.get_or_create_chat_model(
        name="metrics_llm",
        model_type=model_config.type,
        config=model_config,
    )
    llm_adapter = LLMInvokeAdapter(llm)
    
    embedding_config = config.get_language_model_config(config.local_search.embedding_model_id)
    embeddings = model_manager.get_or_create_embedding_model(
        name="metrics_embeddings",
        model_type=embedding_config.type,
        config=embedding_config,
    )

    class EmbeddingsAdapter:
        def __init__(self, embeddings):
            self.embeddings = embeddings
        
        async def aembed_query(self, text):
            def set_model_float(embeddings):
                if hasattr(embeddings, 'force_float32'):
                    try:
                        embeddings.force_float32()
                    except Exception:
                        pass
                model = getattr(embeddings, 'model', None)
                if model is not None and hasattr(model, 'float'):
                    try:
                        model.float()
                    except Exception:
                        pass
            try:
                if hasattr(self.embeddings, 'use_fp16'):
                    self.embeddings.use_fp16 = False
                set_model_float(self.embeddings)
                if hasattr(self.embeddings, 'aembed'):
                    return await self.embeddings.aembed(text)
                elif hasattr(self.embeddings, 'embed'):
                    return self.embeddings.embed(text)
                else:
                    raise AttributeError("No suitable embedding method found in BGEProvider")
            except Exception as e:
                set_model_float(self.embeddings)
                try:
                    if hasattr(self.embeddings, 'aembed'):
                        return await self.embeddings.aembed(text)
                    elif hasattr(self.embeddings, 'embed'):
                        return self.embeddings.embed(text)
                except Exception as e2:
                    raise AttributeError(f"Failed to embed text after float32 retry: {str(e2)}") from e2
                raise AttributeError(f"Failed to embed text: {str(e)}")

    embeddings_adapter = EmbeddingsAdapter(embeddings)

    with open(questions_path, 'r') as f:
        questions = json.load(f)

    novel_contexts = {}
    if input_json_path is not None:
        with open(input_json_path, 'r') as nf:
            novel_data = json.load(nf)
            if isinstance(novel_data, list):
                for entry in novel_data:
                    corpus_name = entry.get('corpus_name')
                    context = entry.get('context', '')
                    if corpus_name:
                        novel_contexts[corpus_name] = context
            elif isinstance(novel_data, dict):
                corpus_name = novel_data.get('corpus_name')
                context = novel_data.get('context', '')
                if corpus_name:
                    novel_contexts[corpus_name] = context

    results = []
    logs = []
    logs_path = os.path.join(os.path.dirname(results_path), LOGS_FILENAME)
    for idx, item in enumerate(tqdm.tqdm(questions, desc="Questions")):
        q = item['question']
        gt = item.get('gold_answer', '')
        source = item.get('source')
        res = {'question': q, 'gold_answer': gt, 'question_type': item.get('question_type', 'Uncategorized')}
        for method in tqdm.tqdm(method_list, desc="Methods", leave=False):
            try:
                answer, context_strs, error_message = await run_method_on_question(
                    method, q, gt, item, config, index_files, novel_contexts, model_manager, llm_adapter, embeddings_adapter, config_path, logs
                )
                if error_message:
                    res[method] = {
                        'answer': error_message,
                        'context': [],
                        'metrics': {}
                    }
                else:
                    metrics = await compute_all_metrics(q, answer, gt, context_strs, llm_adapter, embeddings_adapter, logs, method, log_error_fn=log_error)
                    res[method] = {
                        'answer': answer,
                        'context': [truncate_context(c) for c in context_strs],
                        'metrics': metrics
                    }
            except Exception as e:
                log_error(logs, f'Exception in method {method}: {str(e)}', q, method)
        results.append(res)
        # Batch write every BATCH_SIZE questions
        if (idx + 1) % BATCH_SIZE == 0:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            with open(logs_path, 'w') as logf:
                json.dump(logs, logf, indent=2)
    # Final write at the end
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    with open(logs_path, 'w') as logf:
        json.dump(logs, logf, indent=2)
    print(f"Results saved to {results_path}")

def main() -> None:
    """
    Parse command-line arguments and run the query and evaluation pipeline.
    """
    parser = argparse.ArgumentParser(description="Query GraphRAG and evaluate metrics for a set of questions.")
    parser.add_argument('--project_path', type=str, required=True, help='Path to aime-graphrag project root')
    parser.add_argument('--questions_json', type=str, required=True, help='Path to questions JSON file')
    parser.add_argument('--results_json', type=str, required=True, help='Path to save results JSON')
    parser.add_argument('--methods', type=str, nargs='+', default=DEFAULT_METHODS, help='Search methods to use')
    parser.add_argument('--config', type=str, default=None, help='Path to settings.yaml config file (optional)')
    parser.add_argument('--input_json', type=str, default=None, help='Path to novel.json input document (for llm_with_context baseline)')
    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(args.project_path, 'settings.yaml')
    output_dir = os.path.join(args.project_path, 'output')

    # Set up logging only once
    log_file_path = os.path.join(output_dir, LOGS_FILENAME)
    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    asyncio.run(query_and_evaluate(
        config_path=config_path,
        output_dir=output_dir,
        questions_path=args.questions_json,
        results_path=args.results_json,
        method_list=args.methods,
        project_path=args.project_path,
        input_json_path=args.input_json
    ))

if __name__ == '__main__':
    main()