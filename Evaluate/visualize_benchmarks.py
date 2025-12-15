#!/usr/bin/env python3
"""
Benchmark Visualization Script - Enhanced Edition
Creates beautiful bar charts for bert_score, ragas metrics and pie charts for factual accuracy
and correct/wrong/dont_know counts per question type per model.
Each model has separate bars for each method (basic_search, local_search, etc.)
"""

import os
import json
import math
from numbers import Number
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np
from collections import defaultdict
from pathlib import Path
import argparse
import csv

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')

# Configuration
# Support both host path and container path
import socket
hostname = socket.gethostname()
if os.path.exists("/workspace/Final_Bench_Novel"):
    BENCH_RESULTS_DIR = "/workspace/Final_Bench_Novel"
    OUTPUT_DIR = "/workspace/Final_Bench_Novel/benchmark_visualizations"
else:
    BENCH_RESULTS_DIR = "/home/namit/workspace/Final_Bench_Novel"
    OUTPUT_DIR = "/home/namit/workspace/Final_Bench_Novel/benchmark_visualizations"

# Question types
QUESTION_TYPES = [
    "Fact Retrieval",
    "Complex Reasoning", 
    "Contextual Summarize",
    "Creative Generation"
]

# Modern color palette
COLORS = {
    'primary': '#6366f1',      # Indigo
    'secondary': '#8b5cf6',    # Purple
    'success': '#10b981',      # Emerald
    'warning': '#f59e0b',      # Amber
    'danger': '#ef4444',       # Red
    'info': '#3b82f6',         # Blue
    'dark': '#1f2937',         # Gray 800
    'light': '#f3f4f6',        # Gray 100
}

# Factual accuracy grades with modern colors
GRADES = ["A", "B", "C", "D", "F"]
GRADE_COLORS = {
    "A": "#10b981",  # Emerald
    "B": "#3b82f6",  # Blue
    "C": "#f59e0b",  # Amber
    "D": "#f97316",  # Orange
    "F": "#ef4444"   # Red
}

# Answer type colors
ANSWER_COLORS = {
    "correct": "#10b981",    # Emerald
    "wrong": "#ef4444",      # Red
    "dont_know": "#6b7280"   # Gray
}

# Method colors - modern gradient-inspired palette
METHOD_COLORS = {
    "basic_search": "#3b82f6",      # Blue
    "local_search": "#8b5cf6",      # Purple
    "global_search": "#10b981",     # Emerald
    "llm_with_context": "#f59e0b",  # Amber
    "drift_search": "#ec4899",      # Pink
    "hybrid_search": "#14b8a6",     # Teal
}

# Model colors for comparison charts
MODEL_COLORS = [
    '#6366f1',  # Indigo
    '#ec4899',  # Pink
    '#10b981',  # Emerald
    '#f59e0b',  # Amber
    '#3b82f6',  # Blue
    '#8b5cf6',  # Purple
    '#14b8a6',  # Teal
    '#f97316',  # Orange
]


VALUE_LABEL_EFFECTS = [
    patheffects.withStroke(linewidth=3, foreground='white')
]

BAR_PATH_EFFECTS = [
    patheffects.SimpleLineShadow(offset=(0, -1), shadow_color='#d1d5db'),
    patheffects.Normal()
]


def setup_plot_style():
    """Configure matplotlib for beautiful plots."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#d1d5db',
        'axes.linewidth': 1.5,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#e5e7eb',
        'figure.facecolor': 'white',
        'axes.facecolor': '#fafafa',
        'grid.color': '#e5e7eb',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
    })


def get_method_color(method: str, idx: int = 0) -> str:
    """Get color for a method, with fallback to a color cycle."""
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    return MODEL_COLORS[idx % len(MODEL_COLORS)]


def get_model_color(idx: int) -> str:
    """Get color for a model based on index."""
    return MODEL_COLORS[idx % len(MODEL_COLORS)]


def add_chart_background(ax, fig):
    """Add subtle gradient-like background effect."""
    add_gradient_background(ax)
    style_axes(ax)


def add_gradient_background(ax, top_color: str = '#ffffff', bottom_color: str = '#f3f4f6') -> None:
    """Render a soft vertical gradient behind the axes for visual depth."""
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    gradient = np.hstack([gradient, gradient])
    cmap = LinearSegmentedColormap.from_list('chart-bg', [top_color, bottom_color])
    ax.imshow(
        gradient,
        aspect='auto',
        cmap=cmap,
        extent=[0, 1, 0, 1],
        transform=ax.transAxes,
        zorder=0,
        alpha=0.5
    )
    ax.set_facecolor('none')


def style_axes(ax) -> None:
    """Apply consistent axes styling (grid, spines, tick colors)."""
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine_name, spine in ax.spines.items():
        if spine_name in ('left', 'bottom'):
            spine.set_visible(True)
            spine.set_color('#d1d5db')
            spine.set_linewidth(1.5)
        else:
            spine.set_visible(False)


def annotate_bar_value(ax, bar, value: float, fmt: str = '{value:.1f}', offset: int = 6) -> None:
    """Annotate a single bar with a value label that pops."""
    if value <= 0:
        return
    label = ax.annotate(
        fmt.format(value=value),
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, offset),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold',
        color='#111827'
    )
    label.set_path_effects(VALUE_LABEL_EFFECTS)


def compute_ylim(values: list[float], cap: float = 110.0) -> float:
    """Derive an adaptive y-limit with gentle headroom."""
    clean_values = [v for v in values if v is not None]
    if not clean_values:
        return cap
    max_val = max(clean_values)
    if max_val <= 0:
        return cap
    padded = max_val * 1.1 + 5
    return min(cap, max(padded, 10))


def apply_bar_style(bars) -> None:
    """Give each bar a subtle shadow for depth."""
    for bar in bars:
        bar.set_path_effects(BAR_PATH_EFFECTS)


def is_valid_number(value) -> bool:
    """Return True for real numbers that are not NaN."""
    if not isinstance(value, Number):
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def safe_count(value) -> int:
    """Return a safe integer count, defaulting to 0 for invalid values."""
    if is_valid_number(value):
        return int(value)
    return 0


def load_metrics_data(results_dir: str) -> dict:
    """Load metrics_computed.json from all model folders."""
    data = {}
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist")
        return data
    
    for folder in sorted(os.listdir(results_dir)):
        folder_path = os.path.join(results_dir, folder)
        metrics_file = os.path.join(folder_path, "metrics_computed.json")
        
        if os.path.isdir(folder_path) and os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    # The file contains an array of results
                    if isinstance(metrics_data, list):
                        # Extract model name from folder (e.g., "Mistral_Novel" -> "Mistral")
                        model_name = folder.replace('_Novel', '')
                        data[model_name] = metrics_data
                        print(f"  ✓ Loaded {len(metrics_data)} results from {folder}")
            except Exception as e:
                print(f"  ✗ Error loading {metrics_file}: {e}")
    
    return data


def get_all_methods(data: dict) -> list:
    """Extract all unique methods from the data."""
    methods = set()
    for model_name, metrics_list in data.items():
        for entry in metrics_list:
            method = entry.get("method", "unknown")
            methods.add(method)
    return sorted(list(methods))


def aggregate_metrics_by_model_method(data: dict) -> dict:
    """Aggregate metrics by model and method for bar charts."""
    aggregated = defaultdict(lambda: defaultdict(lambda: {
        "bert_score_f1": [],
        "ragas_score": [],
        "ragas_faithfulness": [],
        "ragas_context_precision": [],
        "ragas_context_recall": [],
        "ragas_answer_relevance": [],
        "semantic_similarity_percentage": []
    }))
    
    for model_name, metrics_list in data.items():
        for entry in metrics_list:
            method = entry.get("method", "unknown")
            metrics = entry.get("metrics", {})
            
            current_metrics = aggregated[model_name][method]
            for key in current_metrics.keys():
                value = metrics.get(key)
                if is_valid_number(value):
                    current_metrics[key].append(float(value))
    
    # Calculate averages
    result = {}
    for model_name in aggregated:
        result[model_name] = {}
        for method in aggregated[model_name]:
            result[model_name][method] = {
                key: np.mean(values) if values else 0 
                for key, values in aggregated[model_name][method].items()
            }
    
    return result


def aggregate_by_question_type_method(data: dict) -> dict:
    """Aggregate metrics by model, question type, and method."""
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "bert_score_f1": [],
        "ragas_score": [],
        "ragas_faithfulness": [],
        "ragas_context_precision": [],
        "ragas_context_recall": [],
        "ragas_answer_relevance": [],
        "factual_accuracy_grades": [],
        "correct_answers_count": 0,
        "wrong_answers_count": 0,
        "dont_know_answers_count": 0
    })))
    
    for model_name, metrics_list in data.items():
        for entry in metrics_list:
            q_type = entry.get("question_type", "Unknown")
            method = entry.get("method", "unknown")
            metrics = entry.get("metrics", {})
            
            current_metrics = aggregated[model_name][q_type][method]
            numeric_keys = [
                "bert_score_f1",
                "ragas_score",
                "ragas_faithfulness",
                "ragas_context_precision",
                "ragas_context_recall",
                "ragas_answer_relevance"
            ]
            for key in numeric_keys:
                value = metrics.get(key)
                if is_valid_number(value):
                    current_metrics[key].append(float(value))
            
            if "factual_accuracy_grade" in metrics and metrics["factual_accuracy_grade"]:
                current_metrics["factual_accuracy_grades"].append(
                    metrics["factual_accuracy_grade"]
                )
            
            current_metrics["correct_answers_count"] += safe_count(metrics.get("correct_answers_count", 0))
            current_metrics["wrong_answers_count"] += safe_count(metrics.get("wrong_answers_count", 0))
            current_metrics["dont_know_answers_count"] += safe_count(metrics.get("dont_know_answers_count", 0))
    
    return aggregated


def export_overall_metrics_to_csv(aggregated_data: dict, methods: list, output_dir: str):
    """Export overall aggregated metrics to CSV file."""
    csv_path = os.path.join(output_dir, "overall_metrics.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Model', 'Method', 'BERT_Score_F1', 'RAGAS_Score', 'RAGAS_Faithfulness', 
                  'RAGAS_Context_Precision', 'RAGAS_Context_Recall', 'RAGAS_Answer_Relevance', 
                  'Semantic_Similarity_Percentage']
        writer.writerow(header)
        
        # Data rows
        for model in sorted(aggregated_data.keys()):
            for method in sorted(methods):
                if method in aggregated_data[model]:
                    metrics = aggregated_data[model][method]
                    row = [
                        model,
                        method,
                        f"{metrics.get('bert_score_f1', 0):.2f}",
                        f"{metrics.get('ragas_score', 0):.2f}",
                        f"{metrics.get('ragas_faithfulness', 0):.2f}",
                        f"{metrics.get('ragas_context_precision', 0):.2f}",
                        f"{metrics.get('ragas_context_recall', 0):.2f}",
                        f"{metrics.get('ragas_answer_relevance', 0):.2f}",
                        f"{metrics.get('semantic_similarity_percentage', 0):.2f}"
                    ]
                    writer.writerow(row)
    
    print(f"  💾 Saved: {csv_path}")


def export_by_question_type_to_csv(by_question_type: dict, methods: list, output_dir: str):
    """Export metrics by question type to CSV file."""
    csv_path = os.path.join(output_dir, "metrics_by_question_type.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Model', 'Question_Type', 'Method', 'BERT_Score_F1', 'RAGAS_Score', 
                  'RAGAS_Faithfulness', 'RAGAS_Context_Precision', 'RAGAS_Context_Recall', 
                  'RAGAS_Answer_Relevance', 'Correct_Answers', 'Wrong_Answers', 
                  'Dont_Know_Answers', 'Total_Questions']
        writer.writerow(header)
        
        # Data rows
        for model in sorted(by_question_type.keys()):
            for q_type in sorted(by_question_type[model].keys()):
                for method in sorted(methods):
                    if method in by_question_type[model][q_type]:
                        metrics = by_question_type[model][q_type][method]
                        
                        # Calculate averages from lists
                        avg_metrics = {}
                        for key in ['bert_score_f1', 'ragas_score', 'ragas_faithfulness',
                                    'ragas_context_precision', 'ragas_context_recall', 
                                    'ragas_answer_relevance']:
                            values = metrics.get(key, [])
                            avg_metrics[key] = np.mean(values) if values else 0
                        
                        correct = metrics.get('correct_answers_count', 0)
                        wrong = metrics.get('wrong_answers_count', 0)
                        dont_know = metrics.get('dont_know_answers_count', 0)
                        total = correct + wrong + dont_know
                        
                        row = [
                            model,
                            q_type,
                            method,
                            f"{avg_metrics['bert_score_f1']:.2f}",
                            f"{avg_metrics['ragas_score']:.2f}",
                            f"{avg_metrics['ragas_faithfulness']:.2f}",
                            f"{avg_metrics['ragas_context_precision']:.2f}",
                            f"{avg_metrics['ragas_context_recall']:.2f}",
                            f"{avg_metrics['ragas_answer_relevance']:.2f}",
                            int(correct),
                            int(wrong),
                            int(dont_know),
                            int(total)
                        ]
                        writer.writerow(row)
    
    print(f"  💾 Saved: {csv_path}")


def export_factual_accuracy_to_csv(by_question_type: dict, methods: list, output_dir: str):
    """Export factual accuracy grade distribution to CSV file."""
    csv_path = os.path.join(output_dir, "factual_accuracy_grades.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Model', 'Question_Type', 'Method', 'Grade_A', 'Grade_B', 'Grade_C', 
                  'Grade_D', 'Grade_F', 'Total_Graded']
        writer.writerow(header)
        
        # Data rows
        from collections import Counter
        for model in sorted(by_question_type.keys()):
            for q_type in sorted(by_question_type[model].keys()):
                for method in sorted(methods):
                    if method in by_question_type[model][q_type]:
                        grades = by_question_type[model][q_type][method].get('factual_accuracy_grades', [])
                        
                        if grades:
                            grade_counts = Counter(grades)
                            total = len(grades)
                            
                            row = [
                                model,
                                q_type,
                                method,
                                grade_counts.get('A', 0),
                                grade_counts.get('B', 0),
                                grade_counts.get('C', 0),
                                grade_counts.get('D', 0),
                                grade_counts.get('F', 0),
                                total
                            ]
                            writer.writerow(row)
    
    print(f"  💾 Saved: {csv_path}")


def create_grouped_bar_chart(aggregated_data: dict, metric_name: str, methods: list, 
                              output_dir: str, title: str = None):
    """Create beautiful grouped bar chart for a specific metric across models with bars for each method."""
    models = list(aggregated_data.keys())
    
    if not models:
        print(f"No data for {metric_name}")
        return
    
    n_methods = len(methods)
    x = np.arange(len(models))
    width = 0.7 / n_methods
    
    fig, ax = plt.subplots(figsize=(14, 8))
    add_chart_background(ax, fig)
    
    plot_values = []
    for i, method in enumerate(methods):
        values = []
        for model in models:
            value = aggregated_data.get(model, {}).get(method, {}).get(metric_name, 0)
            values.append(value)
        plot_values.extend(values)
        
        offset = (i - (n_methods - 1) / 2) * width
        color = get_method_color(method, i)
        bars = ax.bar(x + offset, values, width, label=method.replace('_', ' ').title(), 
                      color=color, edgecolor='white', linewidth=1.5,
                      alpha=0.9, zorder=3)
        apply_bar_style(bars)
        
        # Add value labels on bars with background
        for bar, value in zip(bars, values):
            annotate_bar_value(ax, bar, value)
    
    ax.set_xlabel('Model', fontsize=13, fontweight='bold', color='#374151', labelpad=10)
    ax.set_ylabel('Score', fontsize=13, fontweight='bold', color='#374151', labelpad=10)
    
    # Beautiful title with subtitle effect
    display_title = title or f'{metric_name.replace("_", " ").title()}'
    ax.set_title(display_title, fontsize=18, fontweight='bold', color='#1f2937', pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=11, fontweight='medium')
    
    # Beautiful legend
    legend = ax.legend(title='Search Method', bbox_to_anchor=(1.02, 1), loc='upper left', 
                       fontsize=11, title_fontsize=12, frameon=True, fancybox=True,
                       shadow=True, borderpad=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#e5e7eb')
    
    ax.set_ylim(0, compute_ylim(plot_values))
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('#d1d5db')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"bar_{metric_name}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  📊 Saved: {output_path}")


def create_grouped_bar_chart_by_question_type(aggregated_data: dict, metric_name: str, 
                                               question_type: str, methods: list, output_dir: str):
    """Create beautiful grouped bar chart for a specific metric and question type."""
    models = list(aggregated_data.keys())
    
    # Check if there's data for this question type
    has_data = False
    for model in models:
        for method in methods:
            values = aggregated_data.get(model, {}).get(question_type, {}).get(method, {}).get(metric_name, [])
            if values:
                has_data = True
                break
        if has_data:
            break
    
    if not models or not has_data:
        print(f"  ⚠ No data for {metric_name} - {question_type}")
        return
    
    n_methods = len(methods)
    x = np.arange(len(models))
    width = 0.7 / n_methods
    
    fig, ax = plt.subplots(figsize=(14, 8))
    add_chart_background(ax, fig)
    
    plot_values = []
    for i, method in enumerate(methods):
        values = []
        for model in models:
            metric_values = aggregated_data.get(model, {}).get(question_type, {}).get(method, {}).get(metric_name, [])
            avg_value = np.mean(metric_values) if metric_values else 0
            values.append(avg_value)
        plot_values.extend(values)
        
        offset = (i - (n_methods - 1) / 2) * width
        color = get_method_color(method, i)
        bars = ax.bar(x + offset, values, width, label=method.replace('_', ' ').title(), 
                      color=color, edgecolor='white', linewidth=1.5,
                      alpha=0.9, zorder=3)
        apply_bar_style(bars)
        
        # Add value labels
        for bar, value in zip(bars, values):
            annotate_bar_value(ax, bar, value)
    
    ax.set_xlabel('Model', fontsize=13, fontweight='bold', color='#374151', labelpad=10)
    ax.set_ylabel('Score', fontsize=13, fontweight='bold', color='#374151', labelpad=10)
    
    # Title with question type badge
    ax.set_title(f'{metric_name.replace("_", " ").title()}\n{question_type}', 
                 fontsize=16, fontweight='bold', color='#1f2937', pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=11, fontweight='medium')
    
    legend = ax.legend(title='Search Method', bbox_to_anchor=(1.02, 1), loc='upper left', 
                       fontsize=11, title_fontsize=12, frameon=True, fancybox=True,
                       shadow=True, borderpad=1)
    legend.get_frame().set_facecolor('white')
    
    ax.set_ylim(0, compute_ylim(plot_values))
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    safe_qtype = question_type.replace(" ", "_").lower()
    output_path = os.path.join(output_dir, f"bar_{metric_name}_{safe_qtype}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  📊 Saved: {output_path}")


def create_factual_accuracy_pie_charts(aggregated_data: dict, question_type: str, 
                                        methods: list, output_dir: str):
    """Create beautiful pie charts for factual accuracy grades."""
    models = list(aggregated_data.keys())
    
    if not models:
        return
    
    # Check if there's any data for this question type
    has_data = False
    for model in models:
        for method in methods:
            grades_list = aggregated_data.get(model, {}).get(question_type, {}).get(method, {}).get("factual_accuracy_grades", [])
            if grades_list:
                has_data = True
                break
        if has_data:
            break
    
    if not has_data:
        print(f"  ⚠ No data for factual accuracy - {question_type}")
        return
    
    n_models = len(models)
    n_methods = len(methods)
    
    fig, axes = plt.subplots(n_models, n_methods, figsize=(5*n_methods, 5*n_models))
    fig.patch.set_facecolor('white')
    
    # Handle single model/method case
    if n_models == 1 and n_methods == 1:
        axes = [[axes]]
    elif n_models == 1:
        axes = [axes]
    elif n_methods == 1:
        axes = [[ax] for ax in axes]
    
    for row_idx, model in enumerate(models):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor('white')
            
            grades_list = aggregated_data.get(model, {}).get(question_type, {}).get(method, {}).get("factual_accuracy_grades", [])
            
            if not grades_list:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       fontsize=14, color='#9ca3af', style='italic')
                ax.set_title(f'{model}\n{method.replace("_", " ").title()}', 
                            fontsize=12, fontweight='bold', color='#374151', pad=10)
                ax.axis('off')
                continue
            
            # Count grades
            grade_counts = {grade: 0 for grade in GRADES}
            for grade in grades_list:
                if grade in grade_counts:
                    grade_counts[grade] += 1
            
            # Filter out zero counts
            labels = [g for g in GRADES if grade_counts[g] > 0]
            sizes = [grade_counts[g] for g in labels]
            colors = [GRADE_COLORS[g] for g in labels]
            
            if sizes:
                wedges, texts, autotexts = ax.pie(
                    sizes, labels=labels, colors=colors,
                    autopct='%1.0f%%', startangle=90,
                    textprops={'fontsize': 11, 'fontweight': 'bold'},
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                    explode=[0.02] * len(sizes),
                    shadow=False
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(10)
            
            ax.set_title(f'{model}\n{method.replace("_", " ").title()}', 
                        fontsize=12, fontweight='bold', color='#374151', pad=10)
    
    fig.suptitle(f'Factual Accuracy Grades\n{question_type}', 
                 fontsize=18, fontweight='bold', color='#1f2937', y=1.02)
    plt.tight_layout()
    
    safe_qtype = question_type.replace(" ", "_").lower()
    output_path = os.path.join(output_dir, f"pie_factual_accuracy_{safe_qtype}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  🥧 Saved: {output_path}")


def create_answer_count_pie_charts(aggregated_data: dict, question_type: str, 
                                    methods: list, output_dir: str):
    """Create beautiful pie charts for answer distribution."""
    models = list(aggregated_data.keys())
    
    if not models:
        return
    
    # Check if there's any data for this question type
    has_data = False
    for model in models:
        for method in methods:
            method_data = aggregated_data.get(model, {}).get(question_type, {}).get(method, {})
            total = (method_data.get("correct_answers_count", 0) + 
                     method_data.get("wrong_answers_count", 0) + 
                     method_data.get("dont_know_answers_count", 0))
            if total > 0:
                has_data = True
                break
        if has_data:
            break
    
    if not has_data:
        print(f"  ⚠ No data for answer distribution - {question_type}")
        return
    
    n_models = len(models)
    n_methods = len(methods)
    
    fig, axes = plt.subplots(n_models, n_methods, figsize=(5*n_methods, 5*n_models))
    fig.patch.set_facecolor('white')
    
    # Handle single model/method case
    if n_models == 1 and n_methods == 1:
        axes = [[axes]]
    elif n_models == 1:
        axes = [axes]
    elif n_methods == 1:
        axes = [[ax] for ax in axes]
    
    for row_idx, model in enumerate(models):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor('white')
            
            method_data = aggregated_data.get(model, {}).get(question_type, {}).get(method, {})
            
            correct = method_data.get("correct_answers_count", 0)
            wrong = method_data.get("wrong_answers_count", 0)
            dont_know = method_data.get("dont_know_answers_count", 0)
            
            total = correct + wrong + dont_know
            
            if total == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       fontsize=14, color='#9ca3af', style='italic')
                ax.set_title(f'{model}\n{method.replace("_", " ").title()}', 
                            fontsize=12, fontweight='bold', color='#374151', pad=10)
                ax.axis('off')
                continue
            
            # Prepare data
            labels = []
            sizes = []
            colors = []
            
            if correct > 0:
                labels.append(f'✓ Correct ({int(correct)})')
                sizes.append(correct)
                colors.append(ANSWER_COLORS['correct'])
            if wrong > 0:
                labels.append(f'✗ Wrong ({int(wrong)})')
                sizes.append(wrong)
                colors.append(ANSWER_COLORS['wrong'])
            if dont_know > 0:
                labels.append(f"? Don't Know ({int(dont_know)})")
                sizes.append(dont_know)
                colors.append(ANSWER_COLORS['dont_know'])
            
            if sizes:
                wedges, texts, autotexts = ax.pie(
                    sizes, labels=labels, colors=colors,
                    autopct='%1.0f%%', startangle=90,
                    textprops={'fontsize': 10, 'fontweight': 'bold'},
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                    explode=[0.02] * len(sizes),
                    shadow=False
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            
            ax.set_title(f'{model}\n{method.replace("_", " ").title()}', 
                        fontsize=12, fontweight='bold', color='#374151', pad=10)
    
    fig.suptitle(f'Answer Distribution\n{question_type}', 
                 fontsize=18, fontweight='bold', color='#1f2937', y=1.02)
    plt.tight_layout()
    
    safe_qtype = question_type.replace(" ", "_").lower()
    output_path = os.path.join(output_dir, f"pie_answer_distribution_{safe_qtype}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  🥧 Saved: {output_path}")


def create_comparison_bar_chart(aggregated_data: dict, metric_name: str, 
                                 methods: list, output_dir: str):
    """Create beautiful grouped bar chart comparing models across question types."""
    models = list(aggregated_data.keys())
    question_types = QUESTION_TYPES
    
    # Create one chart per method
    for method in methods:
        # Check if there's data for this method
        has_data = False
        for model in models:
            for q_type in question_types:
                values = aggregated_data.get(model, {}).get(q_type, {}).get(method, {}).get(metric_name, [])
                if values:
                    has_data = True
                    break
            if has_data:
                break
        
        if not has_data:
            continue
            
        x = np.arange(len(question_types))
        width = 0.7 / len(models)
        plot_values = []
        
        fig, ax = plt.subplots(figsize=(14, 8))
        add_chart_background(ax, fig)
        
        for i, model in enumerate(models):
            values = []
            for q_type in question_types:
                metric_values = aggregated_data.get(model, {}).get(q_type, {}).get(method, {}).get(metric_name, [])
                avg_value = np.mean(metric_values) if metric_values else 0
                values.append(avg_value)
            plot_values.extend(values)
            
            offset = (i - (len(models) - 1) / 2) * width
            color = get_model_color(i)
            bars = ax.bar(x + offset, values, width, label=model, 
                          color=color, edgecolor='white', linewidth=1.5,
                          alpha=0.9, zorder=3)
            apply_bar_style(bars)
            
            # Add value labels
            for bar, value in zip(bars, values):
                annotate_bar_value(ax, bar, value, fmt='{value:.0f}')
        
        ax.set_xlabel('Question Type', fontsize=13, fontweight='bold', color='#374151', labelpad=10)
        ax.set_ylabel('Score', fontsize=13, fontweight='bold', color='#374151', labelpad=10)
        ax.set_title(f'{metric_name.replace("_", " ").title()}\n{method.replace("_", " ").title()} Method', 
                     fontsize=16, fontweight='bold', color='#1f2937', pad=15)
        
        ax.set_xticks(x)
        ax.set_xticklabels(question_types, rotation=20, ha='right', fontsize=11, fontweight='medium')
        
        legend = ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', 
                           fontsize=11, title_fontsize=12, frameon=True, fancybox=True,
                           shadow=True, borderpad=1)
        legend.get_frame().set_facecolor('white')
        
        ax.set_ylim(0, compute_ylim(plot_values))
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        safe_method = method.replace(" ", "_").lower()
        output_path = os.path.join(output_dir, f"comparison_{metric_name}_{safe_method}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  📈 Saved: {output_path}")


def create_summary_dashboard(aggregated_data: dict, methods: list, output_dir: str):
    """Create a summary dashboard with key metrics."""
    models = list(aggregated_data.keys())
    
    if not models:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    
    metrics_to_show = [
        ('bert_score_f1', 'BERT Score F1', axes[0, 0]),
        ('ragas_score', 'RAGAS Score', axes[0, 1]),
        ('ragas_faithfulness', 'Faithfulness', axes[1, 0]),
        ('semantic_similarity_percentage', 'Semantic Similarity', axes[1, 1])
    ]
    
    for metric_name, display_name, ax in metrics_to_show:
        add_chart_background(ax, fig)
        
        n_methods = len(methods)
        x = np.arange(len(models))
        width = 0.7 / n_methods
        plot_values = []
        
        for i, method in enumerate(methods):
            values = []
            for model in models:
                value = aggregated_data.get(model, {}).get(method, {}).get(metric_name, 0)
                values.append(value)
            plot_values.extend(values)
            
            offset = (i - (n_methods - 1) / 2) * width
            color = get_method_color(method, i)
            bars = ax.bar(x + offset, values, width, label=method.replace('_', ' ').title(), 
                          color=color, edgecolor='white', linewidth=1, alpha=0.9, zorder=3)
            apply_bar_style(bars)
        
        ax.set_title(display_name, fontsize=14, fontweight='bold', color='#1f2937', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=10)
        ax.set_ylim(0, compute_ylim(plot_values))
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
    
    # Add legend to the figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(methods), 
               fontsize=11, frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.02))
    
    fig.suptitle('Benchmark Results Summary', fontsize=20, fontweight='bold', 
                 color='#1f2937', y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    output_path = os.path.join(output_dir, "summary_dashboard.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  🎯 Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results with beautiful charts')
    parser.add_argument('--input', '-i', type=str, default=BENCH_RESULTS_DIR,
                        help='Input directory containing model benchmark folders')
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_DIR,
                        help='Output directory for visualizations')
    args = parser.parse_args()
    
    input_dir = args.input
    output_dir = args.output
    
    # Setup style
    setup_plot_style()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "═" * 60)
    print("  📊 BENCHMARK VISUALIZATION SUITE")
    print("═" * 60)
    print(f"\n📂 Input:  {input_dir}")
    print(f"📁 Output: {output_dir}")
    print("-" * 60)
    
    # Load data
    print("\n📥 Loading data...")
    data = load_metrics_data(input_dir)
    
    if not data:
        print("\n❌ No data found. Please check the input directory.")
        return
    
    # Get all methods
    methods = get_all_methods(data)
    print(f"\n🔍 Found methods: {', '.join(methods)}")
    print(f"📋 Found models: {', '.join(data.keys())}")
    
    print("\n" + "-" * 60)
    print("🎨 Creating visualizations...")
    print("-" * 60)
    
    # Aggregate data
    overall_aggregated = aggregate_metrics_by_model_method(data)
    by_question_type = aggregate_by_question_type_method(data)
    
    # Export to CSV
    print("\n💾 Exporting data to CSV...")
    export_overall_metrics_to_csv(overall_aggregated, methods, output_dir)
    export_by_question_type_to_csv(by_question_type, methods, output_dir)
    export_factual_accuracy_to_csv(by_question_type, methods, output_dir)
    
    # 1. Summary Dashboard
    print("\n📊 Creating Summary Dashboard...")
    create_summary_dashboard(overall_aggregated, methods, output_dir)
    
    # 2. Overall bar charts for metrics
    print("\n📊 Creating Overall Metric Charts...")
    metrics_to_plot = [
        ("bert_score_f1", "BERT Score F1"),
        ("ragas_score", "RAGAS Score"),
        ("ragas_faithfulness", "RAGAS Faithfulness"),
        ("ragas_context_precision", "Context Precision"),
        ("ragas_context_recall", "Context Recall"),
        ("ragas_answer_relevance", "Answer Relevance"),
        ("semantic_similarity_percentage", "Semantic Similarity")
    ]
    
    for metric_name, title in metrics_to_plot:
        create_grouped_bar_chart(overall_aggregated, metric_name, methods, output_dir, title)
    
    # 3. Bar charts per question type
    print("\n📊 Creating Question Type Charts...")
    question_type_metrics = [
        "bert_score_f1",
        "ragas_score",
        "ragas_faithfulness",
        "ragas_context_precision",
        "ragas_context_recall",
        "ragas_answer_relevance"
    ]
    for q_type in QUESTION_TYPES:
        print(f"\n  📁 {q_type}")
        for metric in question_type_metrics:
            create_grouped_bar_chart_by_question_type(by_question_type, metric, q_type, methods, output_dir)
    
    # 4. Comparison charts
    print("\n📈 Creating Comparison Charts...")
    create_comparison_bar_chart(by_question_type, "bert_score_f1", methods, output_dir)
    create_comparison_bar_chart(by_question_type, "ragas_score", methods, output_dir)
    
    # 5. Pie charts for factual accuracy
    print("\n🥧 Creating Factual Accuracy Pie Charts...")
    for q_type in QUESTION_TYPES:
        create_factual_accuracy_pie_charts(by_question_type, q_type, methods, output_dir)
    
    # 6. Pie charts for answer distribution
    print("\n🥧 Creating Answer Distribution Pie Charts...")
    for q_type in QUESTION_TYPES:
        create_answer_count_pie_charts(by_question_type, q_type, methods, output_dir)
    
    print("\n" + "═" * 60)
    print(f"  ✅ VISUALIZATION COMPLETE!")
    print(f"  📁 Results saved to: {output_dir}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
