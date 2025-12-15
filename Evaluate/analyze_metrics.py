import json
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve, auc
from scipy import stats

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_results(results_path):
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    return data

def create_dataframe(data):
    """Convert results data to DataFrame with metrics extracted."""
    df = pd.DataFrame(data)
    
    # Extract metrics into separate columns
    metrics_df = pd.json_normalize(df['metrics'])
    df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)
    
    return df

def get_configuration_label(method, llm_name=None):
    """Create clear configuration labels for visualization."""
    if method == 'local_search' or method == 'global_search':
        approach = 'GraphRAG'
    elif method == 'basic_search':
        approach = 'RAG'
    elif method == 'direct_llm_with_evidence':
        approach = 'DirectLLM'
    else:
        approach = method.replace('_', ' ').title()
    
    if llm_name:
        return f"{approach}_{llm_name}"
    return approach

def get_configuration_sort_order(configurations):
    """
    Generate a consistent sort order for configurations that works with any number of LLMs.
    Orders by approach first, then by LLM name alphabetically.
    """
    # Define approach priority order
    approach_order = ['GraphRAG', 'RAG', 'DirectLLM']
    
    # Extract unique LLMs and approaches from configurations
    llms = set()
    approaches = set()
    
    for config in configurations:
        for approach in approach_order:
            if config.startswith(approach):
                approaches.add(approach)
                # Extract LLM name (everything after the approach and underscore)
                if '_' in config:
                    llm_name = config.split('_', 1)[1]
                    llms.add(llm_name)
                break
    
    # Sort LLMs alphabetically
    sorted_llms = sorted(list(llms))
    
    # Generate sort order: approach priority × LLM alphabetical
    sort_order = []
    for approach in approach_order:
        if approach in approaches:
            for llm in sorted_llms:
                config_name = f"{approach}_{llm}"
                if config_name in configurations:
                    sort_order.append(config_name)
    
    # Add any remaining configurations that don't match the pattern
    for config in configurations:
        if config not in sort_order:
            sort_order.append(config)
    
    return sort_order

def get_approach_color(approach):
    """Get consistent colors for each approach."""
    colors = {
        'GraphRAG': '#1f77b4',  # Blue
        'RAG': '#ff7f0e',       # Orange
        'DirectLLM': '#2ca02c'  # Green
    }
    return colors.get(approach, '#d62728')

def get_llm_color(llm_name):
    """Get consistent colors for each LLM with support for unlimited LLMs."""
    # Define a color palette that can handle many LLMs
    color_palette = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
        '#aec7e8',  # Light Blue
        '#ffbb78',  # Light Orange
        '#98df8a',  # Light Green
        '#ff9896',  # Light Red
        '#c5b0d5',  # Light Purple
        '#c49c94',  # Light Brown
        '#f7b6d2',  # Light Pink
        '#c7c7c7',  # Light Gray
        '#dbdb8d',  # Light Olive
        '#9edae5',  # Light Cyan
    ]
    
    # Create a mapping for known open-source LLMs with specific colors
    known_llms = {
        'Llama4': '#1f77b4',      # Blue
        'Mistral': '#ff7f0e',     # Orange
        'llama4': '#1f77b4',      # Blue (lowercase)
        'mistral': '#ff7f0e',     # Orange (lowercase)
        'Llama3': '#2ca02c',      # Green
        'llama3': '#2ca02c',      # Green (lowercase)
        'CodeLlama': '#d62728',   # Red
        'codellama': '#d62728',   # Red (lowercase)
        'Phi': '#9467bd',         # Purple
        'phi': '#9467bd',         # Purple (lowercase)
        'Gemma': '#8c564b',       # Brown
        'gemma': '#8c564b',       # Brown (lowercase)
        'Qwen': '#e377c2',        # Pink
        'qwen': '#e377c2',        # Pink (lowercase)
        'Yi': '#7f7f7f',          # Gray
        'yi': '#7f7f7f',          # Gray (lowercase)
        'Falcon': '#bcbd22',      # Olive
        'falcon': '#bcbd22',      # Olive (lowercase)
        'MPT': '#17becf',         # Cyan
        'mpt': '#17becf',         # Cyan (lowercase)
    }
    
    # Return known color if available, otherwise generate from palette
    if llm_name in known_llms:
        return known_llms[llm_name]
    
    # For unknown LLMs, use hash-based color assignment for consistency
    import hashlib
    hash_val = int(hashlib.md5(llm_name.encode()).hexdigest(), 16)
    color_index = hash_val % len(color_palette)
    return color_palette[color_index]

def create_overall_performance_charts(df, output_dir, llm_name=None):
    """
    Create overall performance grouped bar charts for each metric.
    Primary view showing all configurations clearly differentiated.
    """
    plots_dir = os.path.join(output_dir, 'plots', 'overall_performance')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Key metrics to visualize
    key_metrics = ['exact_match', 'f1_score', 'faithfulness', 'hallucination_rate', 
                   'context_relevance', 'triple_f1', 'answer_correctness']
    
    # Filter to available metrics
    available_metrics = [m for m in key_metrics if m in df.columns]
    
    # Create configuration labels
    if 'llm_name' in df.columns:
        df['configuration'] = df.apply(lambda row: get_configuration_label(row['method'], row['llm_name']), axis=1)
    else:
        df['configuration'] = df['method'].apply(lambda m: get_configuration_label(m, llm_name))
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    for metric in available_metrics:
        plt.figure(figsize=(12, 8))
        
        # Group by configuration and calculate mean with error bars
        metric_data = df.groupby('configuration')[metric].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate standard error
        metric_data['se'] = metric_data['std'] / np.sqrt(metric_data['count'])
        
        # Sort configurations for better visualization using dynamic sorting
        config_order = get_configuration_sort_order(metric_data['configuration'].tolist())
        
        # Reorder data
        metric_data = metric_data.set_index('configuration').reindex(config_order).reset_index()
        
        # Create bars with error bars
        bars = plt.bar(range(len(metric_data)), metric_data['mean'], 
                      yerr=metric_data['se'], capsize=5, alpha=0.8)
        
        # Color bars by approach
        for i, (bar, config) in enumerate(zip(bars, metric_data['configuration'])):
            if config.startswith('GraphRAG'):
                bar.set_color('#1f77b4')  # Blue
            elif config.startswith('RAG'):
                bar.set_color('#ff7f0e')  # Orange
            elif config.startswith('DirectLLM'):
                bar.set_color('#2ca02c')  # Green
            else:
                bar.set_color('#d62728')  # Red
        
        # Customize the plot
        plt.title(f'{metric.replace("_", " ").title()} by Configuration', fontsize=16, fontweight='bold')
        plt.xlabel('Configuration', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.xticks(range(len(metric_data)), metric_data['configuration'], rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean_val) in enumerate(zip(bars, metric_data['mean'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'overall_{metric}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Overall performance charts saved to: {plots_dir}")

def create_question_type_nested_charts(df, output_dir, llm_name=None):
    """
    Create nested grouped bar charts showing performance by question type.
    Shows how each configuration performs on different question complexities.
    """
    if 'question_type' not in df.columns:
        print("No question_type column found for nested charts.")
        return
    
    plots_dir = os.path.join(output_dir, 'plots', 'question_type_nested')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Key metrics for question type analysis
    key_metrics = ['f1_score', 'faithfulness', 'exact_match', 'answer_correctness']
    available_metrics = [m for m in key_metrics if m in df.columns]
    
    # Create configuration labels
    if 'llm_name' in df.columns:
        df['configuration'] = df.apply(lambda row: get_configuration_label(row['method'], row['llm_name']), axis=1)
    else:
        df['configuration'] = df['method'].apply(lambda m: get_configuration_label(m, llm_name))
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    for metric in available_metrics:
        plt.figure(figsize=(16, 10))
        
        # Get question types and configurations
        question_types = sorted(df['question_type'].unique())
        configurations = sorted(df['configuration'].unique())
        
        # Prepare data for nested grouping
        plot_data = []
        for qtype in question_types:
            qtype_data = df[df['question_type'] == qtype]
            for config in configurations:
                config_data = qtype_data[qtype_data['configuration'] == config]
                if len(config_data) > 0:
                    mean_val = config_data[metric].mean()
                    std_val = config_data[metric].std()
                    count = len(config_data)
                    se_val = std_val / np.sqrt(count) if count > 0 else 0
                    plot_data.append({
                        'question_type': qtype,
                        'configuration': config,
                        'mean': mean_val,
                        'se': se_val,
                        'count': count
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        if len(plot_df) == 0:
            continue
        
        # Create nested bar chart
        x_pos = np.arange(len(question_types))
        width = 0.8 / len(configurations)  # Adjust width based on number of configurations
        
        for i, config in enumerate(configurations):
            config_data = plot_df[plot_df['configuration'] == config]
            if len(config_data) == 0:
                continue
            
            # Align data with question types
            aligned_data = []
            for qtype in question_types:
                qtype_config_data = config_data[config_data['question_type'] == qtype]
                if len(qtype_config_data) > 0:
                    aligned_data.append(qtype_config_data.iloc[0])
                else:
                    aligned_data.append({'mean': 0, 'se': 0})
            
            means = [d['mean'] for d in aligned_data]
            ses = [d['se'] for d in aligned_data]
            
            # Color by approach
            if config.startswith('GraphRAG'):
                color = '#1f77b4'  # Blue
            elif config.startswith('RAG'):
                color = '#ff7f0e'  # Orange
            elif config.startswith('DirectLLM'):
                color = '#2ca02c'  # Green
            else:
                color = '#d62728'  # Red
            
            bars = plt.bar(x_pos + i * width, means, width, 
                          label=config, color=color, alpha=0.8,
                          yerr=ses, capsize=3)
            
            # Add value labels
            for j, (bar, mean_val) in enumerate(zip(bars, means)):
                if mean_val > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.title(f'{metric.replace("_", " ").title()} by Question Type and Configuration', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Question Type', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.xticks(x_pos + width * (len(configurations) - 1) / 2, question_types, rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'nested_{metric}_by_question_type.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Question type nested charts saved to: {plots_dir}")

def create_trade_off_scatter_plots(df, output_dir, llm_name=None):
    """
    Create scatter plots to investigate relationships between different evaluation aspects.
    """
    plots_dir = os.path.join(output_dir, 'plots', 'trade_off_analysis')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create configuration labels
    if 'llm_name' in df.columns:
        df['configuration'] = df.apply(lambda row: get_configuration_label(row['method'], row['llm_name']), axis=1)
    else:
        df['configuration'] = df['method'].apply(lambda m: get_configuration_label(m, llm_name))
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Define trade-off pairs to analyze
    trade_off_pairs = [
        ('context_relevance', 'f1_score', 'Context Relevance vs F1 Score'),
        ('context_relevance', 'faithfulness', 'Context Relevance vs Faithfulness'),
        ('faithfulness', 'hallucination_rate', 'Faithfulness vs Hallucination Rate'),
        ('answer_correctness', 'f1_score', 'Answer Correctness vs F1 Score'),
        ('triple_precision', 'triple_recall', 'Triple Precision vs Triple Recall')
    ]
    
    for x_metric, y_metric, title in trade_off_pairs:
        if x_metric not in df.columns or y_metric not in df.columns:
            continue
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with different colors for each configuration
        configurations = df['configuration'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(configurations)))
        
        for i, config in enumerate(configurations):
            config_data = df[df['configuration'] == config]
            
            # Remove NaN values
            valid_data = config_data[[x_metric, y_metric]].dropna()
            if len(valid_data) == 0:
                continue
            
            plt.scatter(valid_data[x_metric], valid_data[y_metric], 
                       label=config, color=colors[i], alpha=0.7, s=50)
            
            # Add trend line
            if len(valid_data) > 2:
                z = np.polyfit(valid_data[x_metric], valid_data[y_metric], 1)
                p = np.poly1d(z)
                plt.plot(valid_data[x_metric], p(valid_data[x_metric]), 
                        color=colors[i], alpha=0.5, linestyle='--')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(x_metric.replace('_', ' ').title(), fontsize=12)
        plt.ylabel(y_metric.replace('_', ' ').title(), fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient if possible
        if len(df) > 1:
            corr_data = df[[x_metric, y_metric]].dropna()
            if len(corr_data) > 2:
                correlation = corr_data[x_metric].corr(corr_data[y_metric])
                plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=plt.gca().transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'trade_off_{x_metric}_vs_{y_metric}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Trade-off analysis plots saved to: {plots_dir}")

def create_distribution_box_plots(df, output_dir, llm_name=None):
    """
    Create box plots to show the distribution of scores for each metric across configurations.
    """
    plots_dir = os.path.join(output_dir, 'plots', 'distributions')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create configuration labels
    if 'llm_name' in df.columns:
        df['configuration'] = df.apply(lambda row: get_configuration_label(row['method'], row['llm_name']), axis=1)
    else:
        df['configuration'] = df['method'].apply(lambda m: get_configuration_label(m, llm_name))
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Key metrics for distribution analysis
    key_metrics = ['exact_match', 'f1_score', 'faithfulness', 'hallucination_rate', 
                   'context_relevance', 'answer_correctness']
    available_metrics = [m for m in key_metrics if m in df.columns]
    
    for metric in available_metrics:
        plt.figure(figsize=(14, 8))
        
        # Create box plot
        box_data = [df[df['configuration'] == config][metric].dropna() 
                   for config in sorted(df['configuration'].unique())]
        box_labels = sorted(df['configuration'].unique())
        
        # Remove empty data
        valid_data = [(data, label) for data, label in zip(box_data, box_labels) if len(data) > 0]
        if not valid_data:
            continue
        
        box_data, box_labels = zip(*valid_data)
        
        # Create box plot with custom colors
        colors = []
        for label in box_labels:
            if label.startswith('GraphRAG'):
                colors.append('#1f77b4')  # Blue
            elif label.startswith('RAG'):
                colors.append('#ff7f0e')  # Orange
            elif label.startswith('DirectLLM'):
                colors.append('#2ca02c')  # Green
            else:
                colors.append('#d62728')  # Red
        
        bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title(f'Distribution of {metric.replace("_", " ").title()} by Configuration', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Configuration', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        stats_text = []
        for i, (data, label) in enumerate(zip(box_data, box_labels)):
            mean_val = np.mean(data)
            std_val = np.std(data)
            stats_text.append(f'{label}: μ={mean_val:.3f}, σ={std_val:.3f}')
        
        plt.text(0.02, 0.98, '\n'.join(stats_text), transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'distribution_{metric}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Distribution box plots saved to: {plots_dir}")

def create_triple_evaluation_charts(df, output_dir, llm_name=None):
    """
    Create specific charts for triple evaluation metrics.
    Focus on GraphRAG's triple retrieval and generation capabilities.
    """
    plots_dir = os.path.join(output_dir, 'plots', 'triple_evaluation')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create configuration labels
    if 'llm_name' in df.columns:
        df['configuration'] = df.apply(lambda row: get_configuration_label(row['method'], row['llm_name']), axis=1)
    else:
        df['configuration'] = df['method'].apply(lambda m: get_configuration_label(m, llm_name))
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Triple-specific metrics
    triple_metrics = ['triple_f1', 'triple_precision', 'triple_recall', 'triple_em']
    available_triple_metrics = [m for m in triple_metrics if m in df.columns]
    
    if not available_triple_metrics:
        print("No triple metrics available for triple evaluation charts.")
        return
    
    # 1. Triple metrics bar chart
    plt.figure(figsize=(14, 8))
    
    # Filter to configurations that have triple metrics (mainly GraphRAG)
    triple_configs = []
    for config in df['configuration'].unique():
        config_data = df[df['configuration'] == config]
        if any(config_data[metric].notna().any() for metric in available_triple_metrics):
            triple_configs.append(config)
    
    if not triple_configs:
        print("No configurations with triple metrics found.")
        return
    
    # Prepare data
    x_pos = np.arange(len(triple_configs))
    width = 0.8 / len(available_triple_metrics)
    
    for i, metric in enumerate(available_triple_metrics):
        means = []
        ses = []
        for config in triple_configs:
            config_data = df[df['configuration'] == config]
            metric_data = config_data[metric].dropna()
            if len(metric_data) > 0:
                mean_val = metric_data.mean()
                se_val = metric_data.std() / np.sqrt(len(metric_data))
            else:
                mean_val = 0
                se_val = 0
            means.append(mean_val)
            ses.append(se_val)
        
        bars = plt.bar(x_pos + i * width, means, width, 
                      label=metric.replace('_', ' ').title(), alpha=0.8,
                      yerr=ses, capsize=3)
        
        # Color bars by approach
        for j, (bar, config) in enumerate(zip(bars, triple_configs)):
            if config.startswith('GraphRAG'):
                bar.set_color('#1f77b4')  # Blue
            elif config.startswith('RAG'):
                bar.set_color('#ff7f0e')  # Orange
            elif config.startswith('DirectLLM'):
                bar.set_color('#2ca02c')  # Green
            else:
                bar.set_color('#d62728')  # Red
        
        # Add value labels
        for j, (bar, mean_val) in enumerate(zip(bars, means)):
            if mean_val > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.title('Triple Evaluation Metrics by Configuration', fontsize=16, fontweight='bold')
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(x_pos + width * (len(available_triple_metrics) - 1) / 2, triple_configs, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'triple_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Triple retrieval vs generation comparison (if available)
    if 'triple_retrieval_precision' in df.columns and 'triple_generation_precision' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Filter to GraphRAG configurations
        graphrag_configs = [config for config in triple_configs if config.startswith('GraphRAG')]
        
        if graphrag_configs:
            retrieval_means = []
            generation_means = []
            config_labels = []
            
            for config in graphrag_configs:
                config_data = df[df['configuration'] == config]
                
                retrieval_data = config_data['triple_retrieval_precision'].dropna()
                generation_data = config_data['triple_generation_precision'].dropna()
                
                if len(retrieval_data) > 0 and len(generation_data) > 0:
                    retrieval_means.append(retrieval_data.mean())
                    generation_means.append(generation_data.mean())
                    config_labels.append(config)
            
            if retrieval_means and generation_means:
                x_pos = np.arange(len(config_labels))
                width = 0.35
                
                bars1 = plt.bar(x_pos - width/2, retrieval_means, width, 
                               label='Triple Retrieval Precision', color='#1f77b4', alpha=0.8)
                bars2 = plt.bar(x_pos + width/2, generation_means, width, 
                               label='Triple Generation Precision', color='#ff7f0e', alpha=0.8)
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
                plt.title('GraphRAG: Triple Retrieval vs Generation Precision', fontsize=16, fontweight='bold')
                plt.xlabel('Configuration', fontsize=12)
                plt.ylabel('Precision', fontsize=12)
                plt.xticks(x_pos, config_labels, rotation=45, ha='right')
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                plt.ylim(0, 1.1)
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'triple_retrieval_vs_generation.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    # 3. Triple count analysis (if available)
    if 'triple_count_system' in df.columns and 'triple_count_gold' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Calculate triple coverage ratio
        df['triple_coverage_ratio'] = df['triple_count_system'] / df['triple_count_gold'].replace(0, 1)
        
        # Group by configuration
        coverage_data = df.groupby('configuration')['triple_coverage_ratio'].agg(['mean', 'std', 'count']).reset_index()
        coverage_data['se'] = coverage_data['std'] / np.sqrt(coverage_data['count'])
        
        # Filter to configurations with triple data
        triple_coverage_configs = coverage_data[coverage_data['count'] > 0]
        
        if len(triple_coverage_configs) > 0:
            bars = plt.bar(range(len(triple_coverage_configs)), triple_coverage_configs['mean'], 
                          yerr=triple_coverage_configs['se'], capsize=5, alpha=0.8)
            
            # Color bars by approach
            for i, (bar, config) in enumerate(zip(bars, triple_coverage_configs['configuration'])):
                if config.startswith('GraphRAG'):
                    bar.set_color('#1f77b4')  # Blue
                elif config.startswith('RAG'):
                    bar.set_color('#ff7f0e')  # Orange
                elif config.startswith('DirectLLM'):
                    bar.set_color('#2ca02c')  # Green
                else:
                    bar.set_color('#d62728')  # Red
            
            plt.title('Triple Coverage Ratio by Configuration', fontsize=16, fontweight='bold')
            plt.xlabel('Configuration', fontsize=12)
            plt.ylabel('Triple Coverage Ratio (System/Gold)', fontsize=12)
            plt.xticks(range(len(triple_coverage_configs)), triple_coverage_configs['configuration'], 
                      rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (bar, mean_val) in enumerate(zip(bars, triple_coverage_configs['mean'])):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'triple_coverage_ratio.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Triple evaluation charts saved to: {plots_dir}")

def create_comprehensive_visualizations(df, output_dir, llm_name=None):
    """
    Create all comprehensive visualizations in one function.
    """
    print("Creating comprehensive visualizations...")
    
    # 1. Overall Performance Charts
    create_overall_performance_charts(df, output_dir, llm_name)
    
    # 2. Question Type Nested Charts
    create_question_type_nested_charts(df, output_dir, llm_name)
    
    # 3. Trade-off Analysis
    create_trade_off_scatter_plots(df, output_dir, llm_name)
    
    # 4. Distribution Box Plots
    create_distribution_box_plots(df, output_dir, llm_name)
    
    # 5. Triple Evaluation Charts
    create_triple_evaluation_charts(df, output_dir, llm_name)
    
    # 6. LLM Comparison Summary (if multiple LLMs)
    if 'llm_name' in df.columns and df['llm_name'].nunique() > 1:
        create_llm_comparison_summary(df, output_dir)
    
    print("All comprehensive visualizations completed!")

def create_llm_comparison_summary(df, output_dir):
    """
    Create a comprehensive summary comparing all LLMs across all approaches.
    """
    plots_dir = os.path.join(output_dir, 'plots', 'llm_comparison_summary')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Key metrics for LLM comparison
    key_metrics = ['exact_match', 'f1_score', 'faithfulness', 'hallucination_rate', 
                   'context_relevance', 'answer_correctness']
    available_metrics = [m for m in key_metrics if m in df.columns]
    
    # Create configuration labels
    df['configuration'] = df.apply(lambda row: get_configuration_label(row['method'], row['llm_name']), axis=1)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. LLM Performance Heatmap
    plt.figure(figsize=(16, 10))
    
    # Calculate mean performance for each LLM across all approaches
    llm_performance = df.groupby('llm_name')[available_metrics].mean()
    
    # Create heatmap
    sns.heatmap(llm_performance.T, annot=True, cmap='RdYlGn', center=0.5, fmt='.3f',
                cbar_kws={'label': 'Score'})
    plt.title('LLM Performance Summary Across All Approaches', fontsize=16, fontweight='bold')
    plt.xlabel('LLM', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'llm_performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. LLM Ranking by Approach
    plt.figure(figsize=(14, 8))
    
    # Calculate mean F1 score for each LLM within each approach
    approach_llm_performance = df.groupby(['method', 'llm_name'])['f1_score'].mean().reset_index()
    approach_llm_performance['approach'] = approach_llm_performance['method'].apply(
        lambda x: get_configuration_label(x).split('_')[0] if '_' in get_configuration_label(x) else get_configuration_label(x)
    )
    
    # Create grouped bar chart
    approaches = approach_llm_performance['approach'].unique()
    llms = sorted(approach_llm_performance['llm_name'].unique())
    
    x_pos = np.arange(len(approaches))
    width = 0.8 / len(llms)
    
    for i, llm in enumerate(llms):
        llm_data = approach_llm_performance[approach_llm_performance['llm_name'] == llm]
        means = []
        for approach in approaches:
            approach_data = llm_data[llm_data['approach'] == approach]
            if len(approach_data) > 0:
                means.append(approach_data['f1_score'].iloc[0])
            else:
                means.append(0)
        
        bars = plt.bar(x_pos + i * width, means, width, 
                      label=llm, color=get_llm_color(llm), alpha=0.8)
        
        # Add value labels
        for j, (bar, mean_val) in enumerate(zip(bars, means)):
            if mean_val > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.title('LLM Performance by Approach (F1 Score)', fontsize=16, fontweight='bold')
    plt.xlabel('Approach', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks(x_pos + width * (len(llms) - 1) / 2, approaches, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'llm_ranking_by_approach.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. LLM Consistency Analysis
    plt.figure(figsize=(12, 8))
    
    # Calculate coefficient of variation (std/mean) for each LLM
    llm_consistency = df.groupby('llm_name')['f1_score'].agg(['mean', 'std']).reset_index()
    llm_consistency['cv'] = llm_consistency['std'] / llm_consistency['mean']
    llm_consistency = llm_consistency.sort_values('cv')
    
    bars = plt.bar(range(len(llm_consistency)), llm_consistency['cv'], 
                  color=[get_llm_color(llm) for llm in llm_consistency['llm_name']], alpha=0.8)
    
    plt.title('LLM Consistency Analysis (Coefficient of Variation)', fontsize=16, fontweight='bold')
    plt.xlabel('LLM', fontsize=12)
    plt.ylabel('Coefficient of Variation (Lower = More Consistent)', fontsize=12)
    plt.xticks(range(len(llm_consistency)), llm_consistency['llm_name'], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, cv_val) in enumerate(zip(bars, llm_consistency['cv'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{cv_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'llm_consistency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Best LLM by Question Type
    if 'question_type' in df.columns:
        plt.figure(figsize=(14, 8))
        
        # Calculate mean F1 score for each LLM by question type
        qtype_llm_performance = df.groupby(['question_type', 'llm_name'])['f1_score'].mean().reset_index()
        
        question_types = sorted(qtype_llm_performance['question_type'].unique())
        llms = sorted(qtype_llm_performance['llm_name'].unique())
        
        x_pos = np.arange(len(question_types))
        width = 0.8 / len(llms)
        
        for i, llm in enumerate(llms):
            llm_data = qtype_llm_performance[qtype_llm_performance['llm_name'] == llm]
            means = []
            for qtype in question_types:
                qtype_data = llm_data[llm_data['question_type'] == qtype]
                if len(qtype_data) > 0:
                    means.append(qtype_data['f1_score'].iloc[0])
                else:
                    means.append(0)
            
            bars = plt.bar(x_pos + i * width, means, width, 
                          label=llm, color=get_llm_color(llm), alpha=0.8)
            
            # Add value labels
            for j, (bar, mean_val) in enumerate(zip(bars, means)):
                if mean_val > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.title('LLM Performance by Question Type (F1 Score)', fontsize=16, fontweight='bold')
        plt.xlabel('Question Type', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.xticks(x_pos + width * (len(llms) - 1) / 2, question_types, rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'llm_performance_by_question_type.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"LLM comparison summary charts saved to: {plots_dir}")

def analyze_overall_performance(df):
    """Analyze overall performance by method."""
    print("=== OVERALL PERFORMANCE BY METHOD ===")
    print("=" * 60)
    
    # Define metrics to analyze, including new triple metrics and direct LLM metrics
    metrics_to_analyze = [
        'rouge_score', 'answer_correctness', 'exact_match', 'f1_score',
        'triple_em', 'triple_f1', 'triple_precision', 'triple_recall',
        'context_relevance', 'faithfulness', 'hallucination_rate',
        'context_utilization'  # New metric for direct LLM evaluation
    ]
    
    # Filter to only include metrics that exist in the dataframe
    available_metrics = [m for m in metrics_to_analyze if m in df.columns]
    
    method_stats = df.groupby('method')[available_metrics].agg(['mean', 'std', 'count']).round(4)
    
    print(method_stats)
    print("\n")
    return method_stats

def analyze_question_types(df):
    """Analyze performance by question type."""
    print("\n=== QUESTION TYPE ANALYSIS ===")
    print("=" * 50)
    
    if 'question_type' in df.columns:
        question_types = df['question_type'].unique()
        for q_type in question_types:
            print(f"\n{q_type}:")
            type_data = df[df['question_type'] == q_type]
            
            for method in type_data['method'].unique():
                method_data = type_data[type_data['method'] == method]
                print(f"  {method}:")
                print(f"    - ROUGE: {method_data['rouge_score'].mean():.4f}")
                print(f"    - Correctness: {method_data['answer_correctness'].mean():.4f}")
                print(f"    - F1: {method_data['f1_score'].mean():.4f}")
                print(f"    - Context Relevance: {method_data['context_relevance'].mean():.4f}")
                print(f"    - Faithfulness: {method_data['faithfulness'].mean():.4f}")
                print(f"    - Hallucination: {method_data['hallucination_rate'].mean():.4f}")



def analyze_hallucination(df):
    """Detailed hallucination analysis."""
    print("\n=== HALLUCINATION ANALYSIS ===")
    print("=" * 50)
    
    hallucination_stats = df.groupby('method')['hallucination_rate'].agg(['mean', 'std', 'min', 'max'])
    print("Hallucination Rate Statistics:")
    for method, stats in hallucination_stats.iterrows():
        print(f"  {method}: {stats['mean']:.4f} ± {stats['std']:.4f} (range: {stats['min']:.4f}-{stats['max']:.4f})")
    
    # Count high hallucination cases
    high_hallucination = df[df['hallucination_rate'] > 0.8]
    print(f"\nHigh hallucination cases (>0.8): {len(high_hallucination)}")
    for method in high_hallucination['method'].unique():
        count = len(high_hallucination[high_hallucination['method'] == method])
        print(f"  {method}: {count} cases")

def create_visualizations(df, output_dir):
    """Create visualization plots."""
    try:
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory for plots
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Method comparison heatmap
        metrics_for_plot = ['rouge_score', 'answer_correctness', 'f1_score', 'context_relevance', 'faithfulness']
        plot_data = df.groupby('method')[metrics_for_plot].mean()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(plot_data.T, annot=True, cmap='RdYlGn', center=0.5, fmt='.3f')
        plt.title('Method Performance Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'method_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Hallucination rate comparison
        plt.figure(figsize=(10, 6))
        hallucination_data = df.groupby('method')['hallucination_rate'].mean().sort_values()
        bars = plt.bar(hallucination_data.index, hallucination_data.values, color='red', alpha=0.7)
        plt.title('Hallucination Rate by Method')
        plt.ylabel('Hallucination Rate')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, hallucination_data.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'hallucination_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Metric comparison bar chart
        plt.figure(figsize=(14, 8))
        comparison_data = df.groupby('method')[metrics_for_plot].mean()
        
        x = np.arange(len(comparison_data.index))
        width = 0.15
        
        for i, metric in enumerate(metrics_for_plot):
            plt.bar(x + i*width, comparison_data[metric], width, label=metric.replace('_', ' ').title())
        
        plt.xlabel('Method')
        plt.ylabel('Score')
        plt.title('Metric Comparison by Method')
        plt.xticks(x + width*2, comparison_data.index, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'metric_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualizations saved to: {plots_dir}")
        
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")

def analyze_by_question_type(df):
    """Analyze metrics by question_type and method."""
    print("\n=== ANALYSIS BY QUESTION TYPE ===")
    print("=" * 50)
    if 'question_type' not in df.columns:
        print("No question_type column found.")
        return {}
    
    # Define metrics to analyze, including new triple metrics and direct LLM metrics
    metrics_to_analyze = [
        'rouge_score', 'answer_correctness', 'exact_match', 'f1_score',
        'triple_em', 'triple_f1', 'triple_precision', 'triple_recall',
        'context_relevance', 'faithfulness', 'hallucination_rate',
        'context_utilization'  # New metric for direct LLM evaluation
    ]
    
    # Filter to only include metrics that exist in the dataframe
    available_metrics = [m for m in metrics_to_analyze if m in df.columns]
    
    summary = {}
    for qtype in df['question_type'].unique():
        print(f"\nQuestion Type: {qtype}")
        qtype_df = df[df['question_type'] == qtype]
        stats = qtype_df.groupby('method')[available_metrics].agg(['mean', 'std']).round(4)
        print(stats)
        # Flatten columns for JSON serialization
        stats.columns = ['_'.join(col) for col in stats.columns]
        summary[qtype] = stats.reset_index().to_dict('records')
    return summary

def create_question_type_plots(df, output_dir):
    """Create grouped bar plots for each question_type, showing metrics by method."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    metrics = ['rouge_score', 'answer_correctness', 'f1_score', 'context_relevance', 'faithfulness', 'hallucination_rate']
    plots_dir = os.path.join(output_dir, 'plots', 'question_type')
    os.makedirs(plots_dir, exist_ok=True)
    if 'question_type' not in df.columns:
        print("No question_type column found for plotting.")
        return
    for qtype in df['question_type'].unique():
        qtype_df = df[df['question_type'] == qtype]
        plt.figure(figsize=(12, 7))
        plot_data = qtype_df.groupby('method')[metrics].mean().reset_index()
        plot_data = plot_data.set_index('method')
        plot_data = plot_data[metrics]
        plot_data.plot(kind='bar', ax=plt.gca())
        plt.title(f'Metrics by Method for Question Type: {qtype}')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f'{qtype.replace(" ", "_")}_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot for question type '{qtype}' to {plot_path}")

def get_method_label(method):
    """Map method to user-friendly label."""
    if method == 'local_search':
        return 'GraphRAG (Local Search)'
    elif method == 'global_search':
        return 'GraphRAG (Global Search)'
    elif method == 'basic_search':
        return 'RAG (Basic Search)'
    elif method == 'llm_with_context':
        return 'LLM with Context'
    else:
        return method


def plot_confusion_matrices(df, output_dir, llm_name):
    """Plot confusion matrices for exact_match and answer_correctness (binarized at 0.5).
    Best practice: Clear axis labels, concise titles, colorbar, and colorblind-friendly palette."""
    cm_dir = os.path.join(output_dir, 'plots', 'llm_comparison')
    os.makedirs(cm_dir, exist_ok=True)
    for metric in ['exact_match', 'answer_correctness']:
        for method in df['method'].unique():
            method_label = get_method_label(method)
            y_true = df[df['method'] == method]['gold_answer'].notnull().astype(int)
            if metric == 'exact_match':
                y_pred = (df[df['method'] == method]['exact_match'] > 0.5).astype(int)
            else:
                y_pred = (df[df['method'] == method]['answer_correctness'] > 0.5).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[1,0])
            # Set custom display labels
            if method == 'basic_search':
                display_labels = ['RAG Right', 'RAG Wrong']
            elif method in ['local_search', 'global_search']:
                display_labels = ['GraphRAG Right', 'GraphRAG Wrong']
            else:
                display_labels = ['Right', 'Wrong']
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            fig, ax = plt.subplots(figsize=(5, 5))
            disp.plot(cmap='Blues', ax=ax, colorbar=True)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            plt.title(f'Confusion Matrix: {method_label} ({llm_name}) - {metric}', fontsize=13)
            plt.tight_layout()
            plt.savefig(os.path.join(cm_dir, f'confusion_matrix_{llm_name}_{method}_{metric}.png'), dpi=200)
            plt.close()
    

def plot_overall_qa_performance(df, output_dir, llm_name):
    """Plot overall QA performance (mean exact_match, mean answer_correctness) per method.
    Best practice: Use error bars for variability, clear labels, and colorblind-friendly palette."""
    perf_dir = os.path.join(output_dir, 'plots', 'llm_comparison')
    os.makedirs(perf_dir, exist_ok=True)
    perf_metrics = ['exact_match', 'answer_correctness']
    means = df.groupby('method')[perf_metrics].mean().reset_index()
    means['method_label'] = means['method'].apply(get_method_label)
    for metric in perf_metrics:
        plt.figure(figsize=(8, 5))
        # Use colorblind-friendly palette and rely on errorbar='sd' only
        ax = sns.barplot(x='method_label', y=metric, hue='method_label', data=means, errorbar='sd', palette='colorblind', legend=False)
        plt.title(f'Overall {metric.replace("_", " ").title()} by Method ({llm_name})', fontsize=14)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.xlabel('Method', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=20)
        # Remove legend if it exists (for clarity)
        legend = plt.gca().get_legend()
        if legend is not None:
            legend.remove()
        plt.tight_layout()
        plt.savefig(os.path.join(perf_dir, f'overall_{metric}_{llm_name}.png'), dpi=200)
        plt.close()


def load_and_concat_llm_results(results_paths, llm_names):
    """Load and concatenate results from multiple LLM runs, adding llm_name column."""
    dfs = []
    for path, llm_name in zip(results_paths, llm_names):
        with open(path, 'r') as f:
            data = json.load(f)
        df = create_dataframe(data)
        df['llm_name'] = llm_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def plot_llm_comparison(df, output_dir):
    """Plot metric comparison across LLMs and methods.
    Best practice: Grouped bar plots, error bars, colorblind palette, clear legends."""
    cmp_dir = os.path.join(output_dir, 'plots', 'llm_comparison')
    os.makedirs(cmp_dir, exist_ok=True)
    metrics = ['rouge_score', 'answer_correctness', 'exact_match', 'f1_score', 'context_relevance', 'faithfulness', 'hallucination_rate']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        # Grouped barplot with error bars (standard deviation)
        ax = sns.barplot(
            x=df['method'].apply(get_method_label),
            y=metric,
            hue='llm_name',
            data=df,
            errorbar='sd',
            capsize=0.1,
            palette='colorblind',
        )
        plt.title(f'{metric.replace("_", " ").title()} by Method and LLM', fontsize=14)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.xlabel('Method', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=20)
        plt.legend(title='LLM', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(cmp_dir, f'comparison_{metric}_by_llm.png'), dpi=200)
        plt.close()


def plot_error_distributions(df, output_dir):
    """Plot error distributions for answer_correctness and exact_match per method and LLM.
    Best practice: Use histograms to visualize error patterns for diagnosis."""
    err_dir = os.path.join(output_dir, 'plots', 'llm_comparison')
    os.makedirs(err_dir, exist_ok=True)
    metrics = ['answer_correctness', 'exact_match']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        # Use colorblind-friendly palette
        for i, (llm_name, group) in enumerate(df.groupby('llm_name')):
            for method in group['method'].unique():
                vals = group[group['method'] == method][metric].dropna()
                label = f'{llm_name} - {get_method_label(method)}'
                plt.hist(vals, bins=20, alpha=0.5, label=label)
        plt.title(f'Distribution of {metric.replace("_", " ").title()} by Method and LLM', fontsize=14)
        plt.xlabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(err_dir, f'error_distribution_{metric}_by_llm.png'), dpi=200)
        plt.close()


def plot_roc_pr_curves(df, output_dir, llm_name=None):
    """Plot ROC and Precision-Recall curves for binary metrics (answer_correctness, exact_match, hallucination_rate if available)."""
    import matplotlib.pyplot as plt
    import os
    from sklearn.preprocessing import label_binarize
    roc_dir = os.path.join(output_dir, 'plots', 'roc_pr_curves')
    os.makedirs(roc_dir, exist_ok=True)
    metrics = ['answer_correctness', 'exact_match']
    if 'hallucination_rate' in df.columns or 'hallucination_rate' in df:
        metrics.append('hallucination_rate')
    group_keys = ['method']
    if 'llm_name' in df.columns:
        group_keys.append('llm_name')
    for metric in metrics:
        for keys, group in df.groupby(group_keys):
            if isinstance(keys, str):
                keys = (keys,)
            method = keys[0]
            llm = keys[1] if len(keys) > 1 else llm_name
            y_true = None
            y_score = None
            if metric == 'hallucination_rate':
                # For hallucination, treat faithfulness < 0.5 as hallucination (1), else not (0)
                if 'faithfulness' in group.columns:
                    y_true = (group['faithfulness'] < 0.5).astype(int)
                    y_score = group['hallucination_rate']
                else:
                    continue
            else:
                y_true = (group[metric] > 0.5).astype(int)
                y_score = group[metric]
            if y_true.nunique() < 2:
                continue  # Skip if only one class present
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve: {metric.replace("_", " ").title()}\n{method}{f" ({llm})" if llm else ""}')
            plt.legend(loc='lower right')
            plt.tight_layout()
            fname = f'roc_curve_{metric}_{method}{f"_{llm}" if llm else ""}.png'
            plt.savefig(os.path.join(roc_dir, fname), dpi=200)
            plt.close()
            # PR Curve
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
            plt.figure(figsize=(6, 5))
            plt.plot(recall, precision, label=f'AUC = {pr_auc:.2f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve: {metric.replace("_", " ").title()}\n{method}{f" ({llm})" if llm else ""}')
            plt.legend(loc='lower left')
            plt.tight_layout()
            fname = f'pr_curve_{metric}_{method}{f"_{llm}" if llm else ""}.png'
            plt.savefig(os.path.join(roc_dir, fname), dpi=200)
            plt.close()
    print(f"ROC and PR curves saved to: {roc_dir}")


def plot_metric_dot_table(df, output_dir):
    """Create a table of mean metric scores for each method (and optionally question_type) with embedded bars/dots."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    table_dir = os.path.join(output_dir, 'plots', 'metric_tables')
    os.makedirs(table_dir, exist_ok=True)
    metrics = ['rouge_score', 'answer_correctness', 'exact_match', 'f1_score', 'triple_em', 'triple_f1', 'context_relevance', 'faithfulness', 'hallucination_rate']
    # By method
    means = df.groupby('method')[metrics].mean().reset_index()
    fig, ax = plt.subplots(figsize=(len(metrics) * 1.2 + 2, len(means) * 0.7 + 2))
    ax.axis('off')
    table_vals = means[metrics].values
    norm = plt.Normalize(0, 1)
    table = ax.table(cellText=np.round(table_vals, 3), rowLabels=means['method'], colLabels=[m.replace('_', ' ').title() for m in metrics], loc='center', cellLoc='center')
    # Add embedded bars
    for i in range(table_vals.shape[0]):
        for j in range(table_vals.shape[1]):
            val = table_vals[i, j]
            cell = table[i+1, j]
            cell.get_text().set_text(f'{val:.3f}')
            # Draw a horizontal bar in the cell
            cell._loc = 'center'
            cell.set_facecolor('white')
            bar_width = 0.7 * val  # scale bar width
            bar = plt.Rectangle((0.15, 0.25), bar_width, 0.5, color='tab:blue', alpha=0.3, transform=cell.get_transform(), clip_on=False)
            ax.add_patch(bar)
    plt.title('Mean Metric Scores by Method', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(table_dir, 'metric_dot_table_by_method.png'), dpi=200)
    plt.close()
    # By question_type and method (optional)
    if 'question_type' in df.columns:
        means_qt = df.groupby(['question_type', 'method'])[metrics].mean().reset_index()
        for qtype in means_qt['question_type'].unique():
            qt_means = means_qt[means_qt['question_type'] == qtype]
            fig, ax = plt.subplots(figsize=(len(metrics) * 1.2 + 2, len(qt_means) * 0.7 + 2))
            ax.axis('off')
            table_vals = qt_means[metrics].values
            table = ax.table(cellText=np.round(table_vals, 3), rowLabels=qt_means['method'], colLabels=[m.replace('_', ' ').title() for m in metrics], loc='center', cellLoc='center')
            for i in range(table_vals.shape[0]):
                for j in range(table_vals.shape[1]):
                    val = table_vals[i, j]
                    cell = table[i+1, j]
                    cell.get_text().set_text(f'{val:.3f}')
                    bar_width = 0.7 * val
                    bar = plt.Rectangle((0.15, 0.25), bar_width, 0.5, color='tab:blue', alpha=0.3, transform=cell.get_transform(), clip_on=False)
                    ax.add_patch(bar)
            plt.title(f'Mean Metric Scores by Method\nQuestion Type: {qtype}', fontsize=14)
            plt.tight_layout()
            fname = f'metric_dot_table_{qtype.replace(" ", "_")}.png'
            plt.savefig(os.path.join(table_dir, fname), dpi=200)
            plt.close()
    print(f"Metric dot tables saved to: {table_dir}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze GraphRAG benchmark metrics")
    parser.add_argument('--results_path', type=str, 
                       default='GraphRAG-Bench-Llama4/output/Bench/results.json',
                       help='Path to results.json file')
    parser.add_argument('--output_dir', type=str, 
                       default='GraphRAG-Bench-Llama4/output/Bench',
                       help='Output directory for analysis')
    parser.add_argument('--create_plots', action='store_true',
                       help='Create basic visualization plots')
    parser.add_argument('--create_comprehensive_plots', action='store_true',
                       help='Create comprehensive visualizations (overall performance, question type nested, trade-offs, distributions, triple evaluation)')
    parser.add_argument('--llm_name', type=str, default=None, help='Name of the LLM/model used (e.g., Llama4, Mistral)')
    parser.add_argument('--compare_llms', nargs='+', help='List of results.json files for LLM comparison (use with --llm_names)')
    parser.add_argument('--llm_names', nargs='+', help='List of LLM names for comparison (order must match --compare_llms)')
    args = parser.parse_args()
    
    
    # LLM comparison mode
    if args.compare_llms and args.llm_names:
        df = load_and_concat_llm_results(args.compare_llms, args.llm_names)
        plot_llm_comparison(df, args.output_dir)
        # Also plot confusion matrices and overall QA for each LLM
        for llm_name in args.llm_names:
            df_llm = df[df['llm_name'] == llm_name]
            plot_confusion_matrices(df_llm, args.output_dir, llm_name)
            plot_overall_qa_performance(df_llm, args.output_dir, llm_name)
        # Error analysis: plot error distributions
        plot_error_distributions(df, args.output_dir)
        print(f"LLM comparison plots saved to {os.path.join(args.output_dir, 'plots', 'llm_comparison')}")
        return
    
    # Single LLM mode
    try:
        data = load_results(args.results_path)
        df = create_dataframe(data)
        print(f"Loaded {len(df)} evaluation results")
        print(f"Methods: {', '.join(df['method'].unique())}")
        print(f"Question types: {', '.join(df['question_type'].unique())}")
        print()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run analyses
    method_stats = analyze_overall_performance(df)
    analyze_question_types(df)
    analyze_hallucination(df)
    question_type_summary = analyze_by_question_type(df)

    # Create visualizations if requested
    if args.create_plots:
        create_visualizations(df, args.output_dir)
        if args.llm_name:
            plot_confusion_matrices(df, args.output_dir, args.llm_name)
            plot_overall_qa_performance(df, args.output_dir, args.llm_name)
        # Error analysis: plot error distributions
        plot_error_distributions(df, args.output_dir)
        # New: ROC/PR curves and dot tables
        plot_roc_pr_curves(df, args.output_dir, args.llm_name)
        plot_metric_dot_table(df, args.output_dir)
    
    # Create comprehensive visualizations if requested
    if args.create_comprehensive_plots:
        create_comprehensive_visualizations(df, args.output_dir, args.llm_name)
    
    # Save detailed analysis
    method_stats_dict = {}
    for method in method_stats.index:
        method_stats_dict[method] = {}
        for col in method_stats.columns:
            if isinstance(col, tuple):
                metric_name = col[0]
                stat_name = col[1]
                if metric_name not in method_stats_dict[method]:
                    method_stats_dict[method][metric_name] = {}
                method_stats_dict[method][metric_name][stat_name] = method_stats.loc[method, col]
            else:
                method_stats_dict[method][col] = method_stats.loc[method, col]
    analysis_output = {
        'method_stats': method_stats_dict,
        'question_type_summary': question_type_summary,
        'summary': {
            'total_evaluations': len(df),
            'methods': list(df['method'].unique()),
            'question_types': list(df['question_type'].unique()),
        }
    }
    analysis_path = os.path.join(args.output_dir, 'analysis_summary.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis_output, f, indent=2, cls=NumpyEncoder)
    print(f"\nDetailed analysis saved to: {analysis_path}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 