#!/usr/bin/env python3
"""
Simplified GraphRAG Evaluation Framework

This script provides a clean, modular interface for running comprehensive
evaluations of GraphRAG, RAG, and Direct LLM systems.
"""

import asyncio
import argparse
import sys
from pathlib import Path

from config.evaluation_config import EvaluationConfig
from utils.logging_utils import EvaluationLogger
from core.evaluator import Evaluator


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="GraphRAG Evaluation Framework - Comprehensive evaluation of GraphRAG, RAG, and Direct LLM systems",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--project_path',
        type=str,
        required=True,
        help='Path to aime-graphrag project root'
    )
    
    parser.add_argument(
        '--questions_json',
        type=str,
        required=True,
        help='Path to questions JSON file'
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
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    errors = []
    
    # Check if project path exists
    if not Path(args.project_path).exists():
        errors.append(f"Project path does not exist: {args.project_path}")
    
    # Check if questions file exists
    if not Path(args.questions_json).exists():
        errors.append(f"Questions file does not exist: {args.questions_json}")
    
    # Check if config file exists (if specified)
    if args.config and not Path(args.config).exists():
        errors.append(f"Config file does not exist: {args.config}")
    
    # Check if input JSON exists (if specified)
    if args.input_json and not Path(args.input_json).exists():
        errors.append(f"Input JSON file does not exist: {args.input_json}")
    
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
    else:
        output_dir = Path(args.project_path) / 'output' / 'Bench'
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    return str(output_dir)


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
    else:
        config_path = Path(args.project_path) / 'settings.yaml'
        if not config_path.exists():
            print(f"Warning: Default config file not found: {config_path}")
            config_path = None
    
    # Create configuration
    config = EvaluationConfig(
        llm_model="gpt-4-turbo",  # Default, will be overridden by GraphRAG config
        embedding_model="BAAI/bge-large-en-v1.5",  # Default, will be overridden by GraphRAG config
        judge_llm_model=args.judge_model,  # Use judge model from command line arguments
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
        'project_path': args.project_path,
        'questions_file': args.questions_json,
        'config_file': str(config_path) if config_path else 'None',
        'output_dir': output_dir,
        'methods': args.methods,
        'num_samples': args.num_samples,
        'max_concurrent_tasks': args.max_concurrent_tasks,
        'batch_size': args.batch_size
    })
    
    # Create and run evaluator
    evaluator = Evaluator(config, logger)
    
    try:
        # Run evaluation
        asyncio.run(evaluator.initialize())
        asyncio.run(evaluator.evaluate(args.questions_json))
        
        # Export results if requested
        if args.export_csv:
            evaluator.export_results(output_dir)
        
        print(f"\nEvaluation complete! Results saved to: {output_dir}")
        print(f"  - Results: {config.get_results_path()}")
        print(f"  - Metrics: {config.get_metrics_path()}")
        print(f"  - Logs: {config.get_logs_path()}")
        print(f"  - Query log: {config.get_query_log_path()}")
        
        if args.export_csv:
            print(f"  - CSV export: {output_dir}/results.csv")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        logger.logger.info("Evaluation interrupted by user - saving intermediate results")
        try:
            # Save any intermediate results that exist
            evaluator.save_intermediate_results()
            print(f"Intermediate results saved to: {output_dir}")
            print("You can resume from where you left off or analyze partial results.")
        except Exception as save_error:
            logger.logger.error(f"Failed to save intermediate results: {str(save_error)}")
            print(f"Warning: Could not save intermediate results: {str(save_error)}")
        sys.exit(1)
    except Exception as e:
        logger.logger.error(f"Evaluation failed: {str(e)}")
        print(f"\nEvaluation failed: {str(e)}")
        
        # Try to save intermediate results even on failure
        try:
            evaluator.save_intermediate_results()
            print(f"Partial results saved to: {output_dir}")
            print("Check the logs for error details and partial results.")
        except Exception as save_error:
            logger.logger.error(f"Failed to save intermediate results after error: {str(save_error)}")
            print(f"Warning: Could not save intermediate results: {str(save_error)}")
        
        print("Check the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main() 