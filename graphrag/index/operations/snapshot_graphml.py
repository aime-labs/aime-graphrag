# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing snapshot_graphml method definition."""

import networkx as nx
from typing import Any, Dict, Set

from graphrag.storage.pipeline_storage import PipelineStorage


def _convert_value_for_graphml(value: Any) -> str:
    """Convert any value to a string representation suitable for GraphML.
    
    Handles nested dictionaries, lists, and other complex types by converting
    them to JSON strings. Ensures the result is always a string.
    """
    import json
    from collections.abc import Mapping, Sequence
    
    if value is None:
        return ""
    elif isinstance(value, (str, int, float, bool)):
        return str(value)
    elif isinstance(value, (Mapping, Sequence)) and not isinstance(value, str):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    else:
        return str(value)

def _infer_entity_type(node_name: str) -> str:
    """Infer the entity type based on node name patterns."""
    node_name_lower = node_name.lower()
    
    # Common entity type patterns
    if any(term in node_name_lower for term in ['person', 'prof.', 'dr.', 'mr.', 'mrs.', 'ms.']):
        return 'person'
    elif any(term in node_name_lower for term in ['inc', 'ltd', 'corp', 'llc', 'company']):
        return 'organization'
    elif any(term in node_name_lower for term in ['city', 'country', 'state', 'nation', 'continent']):
        return 'location'
    elif any(term in node_name_lower for term in ['event', 'meeting', 'conference', 'summit']):
        return 'event'
    elif any(term in node_name_lower for term in ['period', 'age', 'era', 'century']):
        return 'time_period'
    return 'concept'

def _generate_description(node_name: str, node_type: str) -> str:
    """Generate a basic description for a node based on its type and name."""
    if node_type == 'person':
        return f"{node_name} is a person."
    elif node_type == 'organization':
        return f"{node_name} is an organization."
    elif node_type == 'location':
        return f"{node_name} is a location."
    elif node_type == 'event':
        return f"{node_name} is an event or meeting."
    elif node_type == 'time_period':
        return f"{node_name} is a time period or era."
    return f"{node_name} is a concept or entity."

def _add_graphml_attributes(graph: nx.Graph) -> None:
    """Add proper attribute types to the graph for GraphML export.
    
    This ensures that all node and edge attributes are properly typed in the GraphML output.
    Converts any complex values to strings to ensure GraphML compatibility.
    """
    # Process node attributes first
    for node, data in graph.nodes(data=True):
        # Preserve existing attributes
        node_data = dict(data)
        
        # Infer or get entity type if not present
        if 'type' not in node_data or not node_data['type']:
            node_data['type'] = _infer_entity_type(node)
        
        # Only generate description if not already present
        if 'description' not in node_data or not node_data['description']:
            node_data['description'] = _generate_description(node, node_data['type'])
        
        # Add label if not present (for visualization)
        if 'label' not in node_data or not node_data['label']:
            node_data['label'] = str(node)
        
        # Update the node data with processed values
        for key, value in node_data.items():
            data[key] = _convert_value_for_graphml(value)
    
    # Process edge attributes
    for u, v, data in graph.edges(data=True):
        # Preserve existing edge data
        edge_data = dict(data)
        
        # Add weight if not present
        if 'weight' not in edge_data:
            edge_data['weight'] = 1.0
            
        # Generate relationship description if not present
        if 'description' not in edge_data or not edge_data['description']:
            source_type = graph.nodes[u].get('type', 'entity')
            target_type = graph.nodes[v].get('type', 'entity')
            edge_data['description'] = f"Relationship between {u} ({source_type}) and {v} ({target_type})"
        
        # Add label if not present
        if 'label' not in edge_data or not edge_data['label']:
            edge_data['label'] = 'related_to'
        
        # Update the edge data with processed values
        for key, value in edge_data.items():
            data[key] = _convert_value_for_graphml(value)
    
    # Process graph attributes
    for key, value in list(graph.graph.items()):
        if isinstance(value, (dict, list, set, tuple)) or value is None:
            graph.graph[key] = _convert_value_for_graphml(value)
    
    # Define required and common attributes
    node_attrs = {
        'type': 'string',
        'description': 'string',
        'label': 'string',
        'source_id': 'string'
    }
    
    edge_attrs = {
        'weight': 'double',
        'description': 'string',
        'label': 'string',
        'source_id': 'string'
    }
    
    # Add any additional attributes found in the graph
    for _, data in graph.nodes(data=True):
        for key in data.keys():
            if key not in node_attrs:
                node_attrs[key] = 'string'  # Default to string type
    
    for _, _, data in graph.edges(data=True):
        for key in data.keys():
            if key not in edge_attrs:
                edge_attrs[key] = 'string'  # Default to string type
    
    # Set graph attributes for GraphML export
    graph.graph.setdefault('node_defaults', {})
    graph.graph.setdefault('edge_defaults', {})
    
    # Set proper attribute types in GraphML
    for attr, attr_type in node_attrs.items():
        if attr not in graph.graph['node_defaults']:
            graph.graph['node_defaults'][attr] = ''
        # Add type information for better schema support
        graph.graph[f'node_{attr}_type'] = attr_type
    
    for attr, attr_type in edge_attrs.items():
        if attr not in graph.graph['edge_defaults']:
            graph.graph['edge_defaults'][attr] = ''
        # Add type information for better schema support
        graph.graph[f'edge_{attr}_type'] = attr_type


def _sanitize_graph_for_export(graph: nx.Graph) -> nx.Graph:
    """Sanitize all graph data to ensure it can be exported to GraphML.
    
    Returns a new graph with all attributes preserved and properly converted to strings.
    """
    # Create a new graph with the same type as the input
    G = type(graph)()
    
    # Add nodes with all original attributes
    for node, data in graph.nodes(data=True):
        # Create a new dictionary for node attributes
        node_attrs = {}
        for key, value in data.items():
            try:
                # Skip None values to avoid serialization issues
                if value is not None:
                    node_attrs[key] = _convert_value_for_graphml(value)
            except Exception as e:
                node_attrs[key] = f"[Error converting {key}: {str(e)}]"
        
        # Add node with all attributes
        G.add_node(node, **node_attrs)
    
    # Add edges with all original attributes
    for u, v, data in graph.edges(data=True):
        # Create a new dictionary for edge attributes
        edge_attrs = {}
        for key, value in data.items():
            try:
                # Skip None values to avoid serialization issues
                if value is not None:
                    edge_attrs[key] = _convert_value_for_graphml(value)
            except Exception as e:
                edge_attrs[key] = f"[Error converting {key}: {str(e)}]"
        
        # Add edge with all attributes
        G.add_edge(u, v, **edge_attrs)
    
    # Add graph attributes
    for key, value in graph.graph.items():
        try:
            if value is not None:
                G.graph[key] = _convert_value_for_graphml(value)
        except Exception as e:
            G.graph[key] = f"[Error converting graph attr {key}: {str(e)}]"
    
    # Ensure required attributes exist
    if not hasattr(G, 'graph'):
        G.graph = {}
    
    return G

async def snapshot_graphml(
    input: str | nx.Graph,
    name: str,
    storage: PipelineStorage,
) -> None:
    """Take a snapshot of a graph in standard GraphML format.
    
    The GraphML output will include all node and edge attributes, including:
    - Node attributes: type, description, source_id
    - Edge attributes: weight, description, source_id
    
    Args:
        input: Either a GraphML string or a NetworkX graph
        name: Base name for the output file (will have .graphml appended)
        storage: Pipeline storage for saving the output
        
    Raises:
        NetworkXError: If there's an error generating the GraphML
    """
    try:
        if isinstance(input, str):
            graphml = input
        else:
            # Create a copy of the graph to avoid modifying the original
            graph = input.copy()
            
            # Add proper GraphML attributes
            _add_graphml_attributes(graph)
            
            # Sanitize the graph for export
            graph = _sanitize_graph_for_export(graph)
            
            # Generate the GraphML content
            graphml = []
            for line in nx.generate_graphml(graph):
                graphml.append(line)
            graphml = "\n".join(graphml)
        
        # Save the GraphML content
        await storage.set(f"{name}.graphml", graphml)
        
    except Exception as e:
        import traceback
        error_msg = f"Failed to generate GraphML: {str(e)}\n{traceback.format_exc()}"
        raise nx.NetworkXError(error_msg) from e
