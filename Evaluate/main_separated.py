#!/usr/bin/env python3
"""
Unified GraphRAG Evaluation Framework with Separated Benchmark and Metrics Processing

This script provides both unified evaluation and separated benchmark/metrics modes:
1. Unified mode: Traditional combined benchmarking and metrics computation
2. Benchmark mode: Only collect raw evaluation data
3. Metrics mode: Only compute metrics from existing benchmark results
"""

import asyncio
import argparse
import sys
import time
from pathlib import Path

from config.evaluation_config import EvaluationConfig
from utils.logging_utils import EvaluationLogger
from core.evaluator import Evaluator
from core.benchmark_runner import BenchmarkRunner
from core.metrics_calculator import MetricsCalculator


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="GraphRAG Evaluation Framework - Comprehensive evaluation with separated benchmark and metrics processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Execution mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['unified', 'benchmark', 'metrics'],
        default='unified',
        help='Execution mode: unified (traditional), benchmark (data collection only), or metrics (computation only)'
    )
    
    # Required arguments for benchmark and unified modes
    parser.add_argument(
        '--project_path',
        type=str,
        help='Path to aime-graphrag project root (required for benchmark and unified modes)'
    )
    
    parser.add_argument(
        '--questions_json',
        type=str,
        help='Path to questions JSON file (required for benchmark and unified modes)'
    )
    
    # Required argument for metrics mode
    parser.add_argument(
        '--benchmark_results',
        type=str,
        help='Path to benchmark results JSON file (required for metrics mode)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to settings.yaml config file (defaults to project_path/settings.yaml)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory to store results (defaults to project_path/output/Bench)'
    )
    
    parser.add_argument(
        '--results_json',
        type=str,
        default="evaluation_results.json",
        help='Filename for results JSON file'
    )
    
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=['local_search', 'global_search', 'basic_search', 'llm_with_context'],
        help='Search methods to use for evaluation'
    )
    
    parser.add_argument(
        '--input_json',
        type=str,
        default=None,
        help='Path to novel.json input document (for llm_with_context baseline)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples per question type to evaluate (optional)'
    )
    
    # Judge model settings
    parser.add_argument(
        '--judge_model',
        type=str,
        default='llama4_chat',
        help='Model to use for LLM-as-a-judge metrics (default: llama4_chat, options: mistral_chat, llama4_chat, gpt_oss_chat)'
    )
    
    parser.add_argument(
        '--judge_base_url',
        type=str,
        default=None,
        help='Base URL for judge model API (optional, defaults to main API URL)'
    )
    
    parser.add_argument(
        '--judge_api_key',
        type=str,
        default=None,
        help='API key for judge model (optional, defaults to main API key)'
    )
    
    # Performance settings
    parser.add_argument(
        '--max_concurrent_tasks',
        type=int,
        default=3,
        help='Maximum number of concurrent evaluation tasks'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size for saving intermediate results'
    )
    
    # Metrics selection for metrics mode
    parser.add_argument(
        '--selected_metrics',
        type=str,
        nargs='+',
        default=None,
        help='Specific metrics to compute (for metrics mode only)'
    )
    
    # Logging settings
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    # Export options
    parser.add_argument(
        '--export_csv',
        action='store_true',
        help='Export results to CSV format'
    )
    
    # Performance optimization flags
    parser.add_argument(
        '--enable_caching',
        action='store_true',
        default=True,
        help='Enable smart caching for LLM and embedding calls'
    )
    
    parser.add_argument(
        '--cache_size',
        type=int,
        default=1000,
        help='Maximum cache size for performance optimization'
    )
    
    # Checkpoint/resume support
    parser.add_argument(
        '--resume_checkpoint',
        action='store_true',
        help='Resume from checkpoint if available (for benchmark and metrics modes)'
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments based on execution mode."""
    errors = []
    
    if args.mode in ['benchmark', 'unified']:
        # Check if project path exists
        if not args.project_path:
            errors.append("--project_path is required for benchmark and unified modes")
        elif not Path(args.project_path).exists():
            errors.append(f"Project path does not exist: {args.project_path}")
        
        # Check if questions file exists
        if not args.questions_json:
            errors.append("--questions_json is required for benchmark and unified modes")
        elif not Path(args.questions_json).exists():
            errors.append(f"Questions file does not exist: {args.questions_json}")
        
        # Check if input JSON exists (if specified)
        if args.input_json and not Path(args.input_json).exists():
            errors.append(f"Input JSON file does not exist: {args.input_json}")
    
    elif args.mode == 'metrics':
        # Check if benchmark results file exists
        if not args.benchmark_results:
            errors.append("--benchmark_results is required for metrics mode")
        elif not Path(args.benchmark_results).exists():
            errors.append(f"Benchmark results file does not exist: {args.benchmark_results}")
    
    # Check if config file exists (if specified)
    if args.config and not Path(args.config).exists():
        errors.append(f"Config file does not exist: {args.config}")
    
    # Validate methods
    valid_methods = ['local_search', 'global_search', 'basic_search', 'llm_with_context']
    for method in args.methods:
        if method not in valid_methods:
            errors.append(f"Invalid method: {method}. Valid methods: {valid_methods}")
    
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


def setup_output_directory(args: argparse.Namespace) -> str:
    """Setup output directory."""
    if args.output_dir:
        output_dir = args.output_dir
    elif args.project_path:
        output_dir = Path(args.project_path) / 'output' / 'Bench'
    else:
        output_dir = './evaluation_output'
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    return str(output_dir)


async def run_unified_mode(args: argparse.Namespace, config: EvaluationConfig, logger: EvaluationLogger):
    """Run traditional unified evaluation mode."""
    print("Running in UNIFIED mode (traditional benchmark + metrics)")
    
    # Create and run evaluator
    evaluator = Evaluator(config, logger)
    
    try:
        # Run evaluation
        await evaluator.initialize()
        await evaluator.evaluate(args.questions_json)
        
        # Export results if requested
        if args.export_csv:
            evaluator.export_results(config.output_dir)
        
        print(f"\\nUnified evaluation complete! Results saved to: {config.output_dir}")
        print(f"  - Results: {config.get_results_path()}")
        print(f"  - Metrics: {config.get_metrics_path()}")
        print(f"  - Logs: {config.get_logs_path()}")
        print(f"  - Query log: {config.get_query_log_path()}")
        
        if args.export_csv:
            print(f"  - CSV export: {config.output_dir}/results.csv")
            
    except Exception as e:
        logger.logger.error(f"Unified evaluation failed: {str(e)}")
        raise
    finally:
        # Always cleanup resources
        await evaluator.cleanup()


async def run_benchmark_mode(args: argparse.Namespace, config: EvaluationConfig, logger: EvaluationLogger):
    """Run benchmark-only mode for data collection."""
    print("Running in BENCHMARK mode (data collection only)")
    
    # Create and run benchmark runner
    benchmark_runner = BenchmarkRunner(config, logger)
    
    try:
        # Run benchmark
        await benchmark_runner.initialize()
        raw_results_path = await benchmark_runner.run_benchmark(
            args.questions_json, 
            resume_from_checkpoint=args.resume_checkpoint
        )
        
        print(f"\\nBenchmark data collection complete!")
        print(f"  - Raw results: {raw_results_path}")
        print(f"  - Summary: {config.output_dir}/benchmark_summary.json")
        print(f"  - Logs: {config.get_logs_path()}")
        print("\\nTo compute metrics, run:")
        print(f"  python main_separated.py --mode metrics --benchmark_results {raw_results_path} --output_dir {config.output_dir}")
        
        return raw_results_path
        
    except Exception as e:
        logger.logger.error(f"Benchmark collection failed: {str(e)}")
        raise
    finally:
        # Always cleanup resources
        await benchmark_runner.cleanup()


async def run_metrics_mode(args: argparse.Namespace, config: EvaluationConfig, logger: EvaluationLogger):
    """Run metrics-only mode for processing existing benchmark results."""
    print("Running in METRICS mode (metrics computation only)")
    
    # Create and run metrics calculator
    metrics_calculator = MetricsCalculator(config, logger)
    
    try:
        # Run metrics computation
        await metrics_calculator.initialize()
        metrics_results_path = await metrics_calculator.compute_metrics_from_benchmark(
            args.benchmark_results, 
            args.selected_metrics,
            resume_from_checkpoint=args.resume_checkpoint
        )
        
        print(f"\\nMetrics computation complete!")
        print(f"  - Metrics results: {metrics_results_path}")
        print(f"  - Summary: {config.output_dir}/metrics_summary.json")
        print(f"  - Logs: {config.get_logs_path()}")
        
        return metrics_results_path
        
    except Exception as e:
        logger.logger.error(f"Metrics computation failed: {str(e)}")
        raise
    finally:
        # Always cleanup resources
        await metrics_calculator.cleanup()


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    # Setup output directory
    output_dir = setup_output_directory(args)
    
    # Determine config path
    if args.config:
        config_path = args.config
    elif args.project_path:
        config_path = Path(args.project_path) / 'settings.yaml'
        if not config_path.exists():
            print(f"Warning: Default config file not found: {config_path}")
            config_path = None
    else:
        config_path = None
    
    # Create configuration
    config = EvaluationConfig(
        llm_model="gpt-4-turbo",  # Default, will be overridden by GraphRAG config
        embedding_model="BAAI/bge-large-en-v1.5",  # Default, will be overridden by GraphRAG config
        judge_llm_model=args.judge_model,
        methods=args.methods,
        output_dir=output_dir,
        results_file=args.results_json,
        max_samples_per_type=args.num_samples,
        project_path=args.project_path,
        config_path=str(config_path) if config_path else None,
        questions_path=args.questions_json,
        input_json_path=args.input_json,
        max_concurrent_tasks=args.max_concurrent_tasks,
        batch_size=args.batch_size
    )
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Setup logging
    logger = EvaluationLogger(output_dir, args.log_level)
    
    # Log system information
    logger.log_system_info({
        'mode': args.mode,
        'project_path': args.project_path or 'N/A',
        'questions_file': args.questions_json or 'N/A',
        'benchmark_results': args.benchmark_results or 'N/A',
        'config_file': str(config_path) if config_path else 'None',
        'output_dir': output_dir,
        'methods': args.methods,
        'num_samples': args.num_samples,
        'max_concurrent_tasks': args.max_concurrent_tasks,
        'batch_size': args.batch_size,
        'selected_metrics': args.selected_metrics or 'All default',
        'resume_checkpoint': args.resume_checkpoint
    })
    
    try:
        start_time = time.time()
        
        # Run based on mode
        if args.mode == 'unified':
            asyncio.run(run_unified_mode(args, config, logger))
        elif args.mode == 'benchmark':
            asyncio.run(run_benchmark_mode(args, config, logger))
        elif args.mode == 'metrics':
            asyncio.run(run_metrics_mode(args, config, logger))
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\\n=== Execution Summary ===")
        print(f"Mode: {args.mode.upper()}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Output directory: {output_dir}")
        
        if args.mode == 'benchmark':
            print("\\nNext steps:")
            print("1. Review the benchmark results in the output directory")
            print("2. Run metrics computation using --mode metrics")
            print("3. Analyze the computed metrics")
        elif args.mode == 'metrics':
            print("\\nNext steps:")
            print("1. Review the computed metrics")
            print("2. Run analysis scripts on the metrics results")
            print("3. Generate visualizations if needed")
        
    except KeyboardInterrupt:
        print("\\nExecution interrupted by user")
        logger.logger.info("Execution interrupted by user - saving intermediate results")
        print("Intermediate results may be available in the output directory.")
        sys.exit(1)
    except Exception as e:
        logger.logger.error(f"Execution failed: {str(e)}")
        print(f"\\nExecution failed: {str(e)}")
        print("Check the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
