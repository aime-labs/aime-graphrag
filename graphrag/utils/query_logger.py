# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Query logging utilities for GraphRAG."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from uuid import uuid4


def _convert_to_serializable(obj):
    """Recursively convert objects to a JSON-serializable format."""
    import pandas as pd
    import numpy as np
    from datetime import datetime, date
    
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        # Convert DataFrame/Series to dict
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        else:
            return obj.to_dict()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return _convert_to_serializable(obj.__dict__)
    else:
        return str(obj)


def log_query(
    output_dir: Union[str, Path],
    query: str,
    method: str,
    response: Any,
    context_data: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
) -> str:
    """
    Log query details to a JSON file in the output directory.
    Maintains a single queries.json file per root directory, appending new queries.

    Args:
        output_dir: Base directory where the queries.json file will be saved/updated
        query: The search query string
        method: The search method used (e.g., 'local', 'global', 'drift', 'basic')
        response: The response from the search
        context_data: Additional context data from the search
        metadata: Additional metadata to include in the log

    Returns:
        str: Path to the saved log file
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use a fixed filename
    log_file = output_path / "queries.json"
    
    # Convert all data to JSON-serializable format
    serialized_context = _convert_to_serializable(context_data)
    serialized_response = _convert_to_serializable(response)
    serialized_metadata = _convert_to_serializable(metadata or {})
    
    # Prepare the query entry with a clean structure
    query_entry = {
        "id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "method": method,
        "metadata": serialized_metadata,
        "api_response": {
            "context_data": serialized_context,
            "response": serialized_response
        }
    }
    
    # Read existing data or initialize empty list
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]  # Convert single object to list if needed
        except (json.JSONDecodeError, Exception):
            data = []
    else:
        data = []
    
    # Append new query entry
    data.append(query_entry)
    
    # Write back to file with consistent formatting
    with open(log_file, 'w', encoding='utf-8') as f:
        # Use a custom encoder to handle datetime objects
        def default_serializer(obj):
            if isinstance(obj, (datetime, Path)):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
        json.dump(
            data, 
            f, 
            indent=2, 
            ensure_ascii=False, 
            default=default_serializer,
            sort_keys=True  # Sort keys for consistent output
        )
    
    return str(log_file)
