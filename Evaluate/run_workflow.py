#!/usr/bin/env python3
"""
Convenience script for running separated benchmark and metrics computation phases efficiently.
This script provides optimized workflows for different use cases.
"""

import asyncio
import argparse
import sys
import time
from pathlib import Path

from main_separated import (
    create_parser, validate_arguments, setup_output_directory,
    run_benchmark_mode, run_metrics_mode
)
from config.evaluation_config import EvaluationConfig
from utils.logging_utils import EvaluationLogger


def create_workflow_parser() -> argparse.ArgumentParser:
    """Create parser for workflow-specific arguments."""
    parser = argparse.ArgumentParser(
        description="GraphRAG Evaluation Workflow - Efficient separated benchmark and metrics processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Workflow selection
    parser.add_argument(
        '--workflow',
        type=str,
        choices=['fast_benchmark', 'comprehensive_metrics', 'full_separated', 'resume_metrics'],
        default='full_separated',
        help='Workflow type: fast_benchmark (optimize for speed), comprehensive_metrics (all metrics), full_separated (both phases), resume_metrics (metrics only)'
    )
    
    # Base arguments (copied from main parser)
    parser.add_argument('--project_path', type=str, required=True, help='Path to aime-graphrag project root')
    parser.add_argument('--questions_json', type=str, required=True, help='Path to questions JSON file')
    parser.add_argument('--config', type=str, default=None, help='Path to settings.yaml config file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--methods', type=str, nargs='+', default=['local_search', 'global_search', 'basic_search', 'llm_with_context'], help='Search methods')
    parser.add_argument('--input_json', type=str, default=None, help='Path to novel.json input document')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples per question type')
    
    # Judge model settings
    parser.add_argument('--judge_model', type=str, default='llama4_chat', help='Judge model (options: mistral_chat, llama4_chat, gpt_oss_chat)')
    parser.add_argument('--judge_base_url', type=str, default=None, help='Judge model base URL')
    parser.add_argument('--judge_api_key', type=str, default=None, help='Judge model API key')
    
    # Performance optimization settings
    parser.add_argument('--max_concurrent_tasks', type=int, default=3, help='Max concurrent tasks (higher for separated mode)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (higher for separated mode)')
    parser.add_argument('--cache_size', type=int, default=2000, help='Cache size for optimization')
    
    # Workflow-specific settings
    parser.add_argument(
        '--fast_mode',
        action='store_true',
        help='Enable fast mode optimizations (may slightly reduce accuracy for speed)'
    )
    
    parser.add_argument(
        '--selected_metrics',
        type=str,
        nargs='+',
        default=None,
        help='Specific metrics to compute (for metrics-only workflows)'
    )
    
    parser.add_argument(
        '--resume_from_benchmark',
        type=str,
        default=None,
        help='Path to existing benchmark results to resume from'
    )
    
    # Logging and output
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--export_csv', action='store_true', help='Export results to CSV')
    parser.add_argument('--create_visualizations', action='store_true', help='Create analysis visualizations')
    
    return parser


def optimize_config_for_workflow(config: EvaluationConfig, workflow: str, fast_mode: bool) -> EvaluationConfig:
    """Optimize configuration based on workflow type."""
    
    if workflow == 'fast_benchmark':
        # Optimize for speed in benchmark collection
        config.max_concurrent_tasks = min(config.max_concurrent_tasks * 2, 10)
        config.batch_size = min(config.batch_size * 2, 8)
        
    elif workflow == 'comprehensive_metrics':
        # Optimize for accuracy in metrics computation
        config.max_concurrent_tasks = max(config.max_concurrent_tasks // 2, 3)
        config.batch_size = max(config.batch_size // 2, 2)
        
    elif workflow == 'full_separated':
        # Balanced optimization for both phases
        config.max_concurrent_tasks = config.max_concurrent_tasks
        config.batch_size = config.batch_size
    
    if fast_mode:
        # Additional optimizations for fast mode
        config.max_concurrent_tasks = int(min(config.max_concurrent_tasks * 1.5, 12))
        config.batch_size = int(min(config.batch_size * 1.5, 10))
    
    return config


async def run_fast_benchmark_workflow(args, config: EvaluationConfig, logger: EvaluationLogger):
    """Run optimized benchmark collection workflow."""
    print("🚀 Running FAST BENCHMARK workflow")
    print("   - Optimized for speed")
    print("   - Higher concurrency")
    print("   - Aggressive caching")
    print()
    
    start_time = time.time()
    
    # Run benchmark with fast optimizations
    raw_results_path = await run_benchmark_mode(args, config, logger)
    
    duration = time.time() - start_time
    
    print(f"\n✅ Fast benchmark complete in {duration:.2f} seconds!")
    print(f"📁 Raw results: {raw_results_path}")
    print("\n🔄 Next step: Run metrics computation")
    print(f"   python run_workflow.py --workflow comprehensive_metrics --resume_from_benchmark {raw_results_path}")
    
    return raw_results_path


async def run_comprehensive_metrics_workflow(args, config: EvaluationConfig, logger: EvaluationLogger):
    """Run comprehensive metrics computation workflow."""
    print("📊 Running COMPREHENSIVE METRICS workflow")
    print("   - All available metrics")
    print("   - Optimized for accuracy")
    print("   - Advanced caching")
    print()
    
    # Determine benchmark results path
    if args.resume_from_benchmark:
        benchmark_results_path = args.resume_from_benchmark
    else:
        # Look for existing benchmark results
        benchmark_results_path = Path(config.output_dir) / "benchmark_raw_results.json"
        if not benchmark_results_path.exists():
            print("❌ No benchmark results found!")
            print("   Run benchmark first or specify --resume_from_benchmark")
            return None
    
    print(f"📂 Loading benchmark results from: {benchmark_results_path}")
    
    start_time = time.time()
    
    # Create a mock args object for metrics mode
    metrics_args = argparse.Namespace()
    metrics_args.benchmark_results = str(benchmark_results_path)
    metrics_args.selected_metrics = args.selected_metrics
    
    # Run metrics computation
    metrics_results_path = await run_metrics_mode(metrics_args, config, logger)
    
    duration = time.time() - start_time
    
    print(f"\n✅ Comprehensive metrics complete in {duration:.2f} seconds!")
    print(f"📊 Metrics results: {metrics_results_path}")
    
    if args.create_visualizations:
        print("\n📈 Creating analysis visualizations...")
        try:
            # Try to import and use existing visualization tools
            from benchmark_visualization import create_visualization_dashboard
            create_visualization_dashboard(config.output_dir)
            print("   ✓ Visualizations created using benchmark_visualization.py")
        except ImportError:
            print("   ⚠️  Visualization dependencies not available")
            print("   💡 Install visualization dependencies or run: python benchmark_visualization.py")
        except Exception as e:
            print(f"   ⚠️  Visualization creation failed: {e}")
            print("   💡 Run manually: python benchmark_visualization.py --output_dir {config.output_dir}")
    
    return metrics_results_path


async def run_full_separated_workflow(args, config: EvaluationConfig, logger: EvaluationLogger):
    """Run complete separated workflow (benchmark + metrics)."""
    print("🔄 Running FULL SEPARATED workflow")
    print("   - Phase 1: Benchmark data collection")
    print("   - Phase 2: Metrics computation")
    print("   - Optimized for both speed and accuracy")
    print()
    
    total_start_time = time.time()
    
    # Phase 1: Benchmark
    print("=" * 50)
    print("📊 PHASE 1: BENCHMARK DATA COLLECTION")
    print("=" * 50)
    
    benchmark_start = time.time()
    raw_results_path = await run_benchmark_mode(args, config, logger)
    benchmark_duration = time.time() - benchmark_start
    
    print(f"\n✅ Phase 1 complete in {benchmark_duration:.2f} seconds!")
    print(f"📁 Raw results: {raw_results_path}")
    
    # Phase 2: Metrics
    print("\n" + "=" * 50)
    print("📈 PHASE 2: METRICS COMPUTATION")
    print("=" * 50)
    
    metrics_start = time.time()
    
    # Create metrics args
    metrics_args = argparse.Namespace()
    metrics_args.benchmark_results = raw_results_path
    metrics_args.selected_metrics = args.selected_metrics
    
    metrics_results_path = await run_metrics_mode(metrics_args, config, logger)
    metrics_duration = time.time() - metrics_start
    
    total_duration = time.time() - total_start_time
    
    print(f"\n✅ Phase 2 complete in {metrics_duration:.2f} seconds!")
    print(f"📊 Metrics results: {metrics_results_path}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 FULL WORKFLOW COMPLETE!")
    print("=" * 50)
    print(f"⏱️  Total time: {total_duration:.2f} seconds")
    print(f"📊 Benchmark time: {benchmark_duration:.2f} seconds")
    print(f"📈 Metrics time: {metrics_duration:.2f} seconds")
    print(f"⚡ Time saved by separation: ~{(benchmark_duration + metrics_duration - total_duration):.1f}s")
    
    if args.export_csv:
        print("\n📄 Exporting to CSV...")
        try:
            # Try to export using existing result processor functionality
            from core.result_processor import ResultProcessor
            from utils.data_utils import DataProcessor
            
            processor = ResultProcessor(logger, DataProcessor(logger.logger))
            
            # Try to load results and export to CSV
            import json
            results_file = Path(config.output_dir) / "evaluation_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                processor.results = results if isinstance(results, list) else [results]
                processor.export_to_csv(config.output_dir)
                print("   ✓ CSV files created in output directory")
            else:
                print("   ⚠️  No results file found to export")
        except Exception as e:
            print(f"   ⚠️  CSV export failed: {e}")
            print("   💡 Check results files and try manual export")
    
    if args.create_visualizations:
        print("\n📈 Creating visualizations...")
        try:
            # Try to create visualizations using existing tools
            from benchmark_visualization import main as create_viz
            import sys
            old_argv = sys.argv
            sys.argv = ['benchmark_visualization.py', '--output_dir', str(config.output_dir)]
            create_viz()
            sys.argv = old_argv
            print("   ✓ Visualizations created using benchmark_visualization.py")
        except ImportError:
            print("   ⚠️  Visualization dependencies not available")
        except Exception as e:
            print(f"   ⚠️  Visualization creation failed: {e}")
            print(f"   💡 Run manually: python benchmark_visualization.py --output_dir {config.output_dir}")
    
    print("\n🔍 Analysis suggestions:")
    print(f"   1. Review results: {config.output_dir}")
    print(f"   2. Analyze metrics: python analyze_metrics.py --results_path {metrics_results_path}")
    print(f"   3. Compare methods: python benchmark_visualization.py --output_dir {config.output_dir}")
    
    return raw_results_path, metrics_results_path


def main():
    """Main workflow entry point."""
    parser = create_workflow_parser()
    args = parser.parse_args()
    
    # Validate basic arguments
    if not Path(args.project_path).exists():
        print(f"❌ Project path does not exist: {args.project_path}")
        sys.exit(1)
    
    if not Path(args.questions_json).exists():
        print(f"❌ Questions file does not exist: {args.questions_json}")
        sys.exit(1)
    
    # Setup output directory
    output_dir = setup_output_directory(args)
    
    # Determine config path
    if args.config:
        config_path = args.config
    else:
        config_path = Path(args.project_path) / 'settings.yaml'
        if not config_path.exists():
            print(f"⚠️  Warning: Default config file not found: {config_path}")
            config_path = None
    
    # Create configuration
    config = EvaluationConfig(
        llm_model="gpt-4-turbo",
        embedding_model="BAAI/bge-large-en-v1.5",
        judge_llm_model=args.judge_model,
        methods=args.methods,
        output_dir=output_dir,
        results_file="evaluation_results.json",
        max_samples_per_type=args.num_samples,
        project_path=args.project_path,
        config_path=str(config_path) if config_path else None,
        questions_path=args.questions_json,
        input_json_path=args.input_json,
        max_concurrent_tasks=args.max_concurrent_tasks,
        batch_size=args.batch_size
    )
    
    # Optimize config for workflow
    config = optimize_config_for_workflow(config, args.workflow, args.fast_mode)
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)
    
    # Setup logging
    logger = EvaluationLogger(output_dir, args.log_level)
    
    # Log workflow information
    logger.log_system_info({
        'workflow': args.workflow,
        'fast_mode': args.fast_mode,
        'project_path': args.project_path,
        'questions_file': args.questions_json,
        'config_file': str(config_path) if config_path else 'None',
        'output_dir': output_dir,
        'methods': args.methods,
        'num_samples': args.num_samples,
        'max_concurrent_tasks': config.max_concurrent_tasks,
        'batch_size': config.batch_size,
        'selected_metrics': args.selected_metrics or 'All default'
    })
    
    print("🚀 GraphRAG Evaluation Workflow")
    print(f"📋 Workflow: {args.workflow.upper()}")
    print(f"⚡ Fast mode: {'ON' if args.fast_mode else 'OFF'}")
    print(f"🔧 Concurrency: {config.max_concurrent_tasks} tasks")
    print(f"📦 Batch size: {config.batch_size}")
    print(f"📂 Output: {output_dir}")
    print()
    
    try:
        # Run selected workflow
        if args.workflow == 'fast_benchmark':
            asyncio.run(run_fast_benchmark_workflow(args, config, logger))
        elif args.workflow == 'comprehensive_metrics':
            asyncio.run(run_comprehensive_metrics_workflow(args, config, logger))
        elif args.workflow == 'full_separated':
            asyncio.run(run_full_separated_workflow(args, config, logger))
        elif args.workflow == 'resume_metrics':
            asyncio.run(run_comprehensive_metrics_workflow(args, config, logger))
        
    except KeyboardInterrupt:
        print("\n⏹️  Workflow interrupted by user")
        print("💾 Intermediate results may be available in the output directory")
        sys.exit(1)
    except Exception as e:
        logger.logger.error(f"Workflow failed: {str(e)}")
        print(f"\n❌ Workflow failed: {str(e)}")
        print("📋 Check the logs for more details")
        sys.exit(1)


if __name__ == "__main__":
    main()
