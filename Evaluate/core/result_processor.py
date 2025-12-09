import json
import asyncio
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from utils.data_utils import DataProcessor
from utils.logging_utils import EvaluationLogger


class ResultProcessor:
    """Process and manage evaluation results."""
    
    def __init__(self, logger: EvaluationLogger, data_processor: DataProcessor):
        self.logger = logger
        self.data_processor = data_processor
        self.results: List[Dict[str, Any]] = []
        self.metrics: List[Dict[str, Any]] = []
        self.embeddings: List[Dict[str, Any]] = []
        self.judge_evaluations: List[Dict[str, Any]] = []
        self.query_log: List[Dict[str, Any]] = []
    
    def add_result(self, result: Dict[str, Any]):
        """Add a single evaluation result and extract metrics."""
        # Extract metrics from result to store separately
        if 'metrics' in result:
            metric_result = self.data_processor.create_metric_result(
                {'id': result.get('question_id'),
                 'question': result.get('question'),
                 'question_type': result.get('question_type'),
                 'source': result.get('source')},
                result.get('method', ''),
                result.get('metrics', {}),
                result.get('error_message', '')
            )
            # Add computation_time if present in result
            if 'computation_time' in result:
                metric_result['computation_time'] = result['computation_time']
            self.metrics.append(metric_result)
        
        # Store result without metrics
        result_copy = result.copy()
        if 'metrics' in result_copy:
            del result_copy['metrics']  # Remove metrics to avoid duplication
        # Also remove computation_time as it's now in metrics.json
        if 'computation_time' in result_copy:
            del result_copy['computation_time']
        self.results.append(result_copy)
    
    def add_query_log(self, query_entry: Dict[str, Any]):
        """Add a query log entry."""
        self.query_log.append(query_entry)
    
    def add_embedding(self, embedding_entry: Dict[str, Any]):
        """Add an embedding entry."""
        self.embeddings.append(embedding_entry)
    
    def add_judge_evaluation(self, judge_entry: Dict[str, Any]):
        """Add a judge evaluation entry."""
        self.judge_evaluations.append(judge_entry)
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all results."""
        return self.results.copy()
    
    def get_query_log(self) -> List[Dict[str, Any]]:
        """Get all query log entries."""
        return self.query_log.copy()
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get all metrics."""
        return self.metrics.copy()
    
    def get_embeddings(self) -> List[Dict[str, Any]]:
        """Get all embeddings."""
        return self.embeddings.copy()
    
    def get_judge_evaluations(self) -> List[Dict[str, Any]]:
        """Get all judge evaluations."""
        return self.judge_evaluations.copy()
    
    def save_results(self, output_dir: str, filename: str = "results.json"):
        """Save results to file."""
        output_path = Path(output_dir) / filename
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.logger.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.logger.error(f"Failed to save results: {str(e)}")
            raise

    def save_metrics(self, output_dir: str, filename: str = "metrics.json"):
        """Save metrics to file."""
        output_path = Path(output_dir) / filename
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            
            self.logger.logger.info(f"Metrics saved to {output_path}")
            
        except Exception as e:
            self.logger.logger.error(f"Failed to save metrics: {str(e)}")
            raise
    
    def save_query_log(self, output_dir: str, filename: str = "query.json"):
        """Save query log to file."""
        output_path = Path(output_dir) / filename
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.query_log, f, indent=2, default=str)
            
            self.logger.logger.info(f"Query log saved to {output_path}")
            
        except Exception as e:
            self.logger.logger.error(f"Failed to save query log: {str(e)}")
            raise
    
    def save_embeddings(self, output_dir: str, filename: str = "embeddings.json"):
        """Save embeddings to file."""
        output_path = Path(output_dir) / filename
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.embeddings, f, indent=2, default=str)
            
            self.logger.logger.info(f"Embeddings saved to {output_path}")
            
        except Exception as e:
            self.logger.logger.error(f"Failed to save embeddings: {str(e)}")
            raise
    
    def save_judge_evaluations(self, output_dir: str, filename: str = "judge.json"):
        """Save judge evaluations to file."""
        output_path = Path(output_dir) / filename
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.judge_evaluations, f, indent=2, default=str)
            
            self.logger.logger.info(f"Judge evaluations saved to {output_path}")
            
        except Exception as e:
            self.logger.logger.error(f"Failed to save judge evaluations: {str(e)}")
            raise
    
    def save_benchmark_summary(self, output_dir: str, config: Any, filename: str = "benchmark_summary.json"):
        """Save benchmark summary for unified mode (matches separated mode format)."""
        from datetime import datetime, timezone
        output_path = Path(output_dir) / filename
        
        try:
            summary = {
                'total_results': len(self.results),
                'methods': list(set(r.get('method', 'unknown') for r in self.results)),
                'question_types': list(set(r.get('question_type', 'unknown') for r in self.results)),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'config': {
                    'max_concurrent_tasks': config.max_concurrent_tasks,
                    'batch_size': config.batch_size,
                    'cache_enabled': True
                }
            }
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.logger.info(f"Benchmark summary saved to {output_path}")
            
        except Exception as e:
            self.logger.logger.error(f"Failed to save benchmark summary: {str(e)}")
            raise
    
    def save_metrics_summary(self, output_dir: str, config: Any, filename: str = "metrics_summary.json"):
        """Save metrics summary for unified mode (matches separated mode format)."""
        from datetime import datetime, timezone
        import numpy as np
        output_path = Path(output_dir) / filename
        
        try:
            # Collect all unique metrics
            all_metrics = set()
            for metric_entry in self.metrics:
                if 'metrics' in metric_entry and isinstance(metric_entry['metrics'], dict):
                    all_metrics.update(metric_entry['metrics'].keys())
            
            # Aggregate metrics
            aggregated = {}
            for metric_name in all_metrics:
                values = []
                for metric_entry in self.metrics:
                    if 'metrics' in metric_entry and isinstance(metric_entry['metrics'], dict):
                        val = metric_entry['metrics'].get(metric_name)
                        if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                            values.append(val)
                
                if values:
                    aggregated[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values)
                    }
            
            summary = {
                'total_metrics_computed': len(self.metrics),
                'metrics_aggregated': aggregated,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'config': {
                    'max_concurrent_tasks': config.max_concurrent_tasks,
                    'batch_size': config.batch_size
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.logger.info(f"Metrics summary saved to {output_path}")
            
        except Exception as e:
            self.logger.logger.error(f"Failed to save metrics summary: {str(e)}")
            raise
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary from results."""
        # Reconstruct full results with metrics for summary generation
        full_results = []
        for i, result in enumerate(self.results):
            if i < len(self.metrics):
                full_result = result.copy()
                full_result['metrics'] = self.metrics[i]['metrics']
                full_results.append(full_result)
            else:
                full_results.append(result)
        return self.data_processor.get_performance_summary(full_results)
    
    def get_results_by_type(self, question_type: str) -> List[Dict[str, Any]]:
        """Get results filtered by question type."""
        return self.data_processor.filter_results_by_type(self.results, question_type)
    
    def get_results_by_method(self, method: str) -> List[Dict[str, Any]]:
        """Get results filtered by method."""
        return self.data_processor.filter_results_by_method(self.results, method)
    
    def get_aggregated_metrics(self, question_type: Optional[str] = None, method: Optional[str] = None) -> Dict[str, float]:
        """Get aggregated metrics for filtered results."""
        filtered_metrics = self.metrics
        
        if question_type:
            filtered_metrics = [m for m in filtered_metrics if m.get('question_type') == question_type]
        
        if method:
            filtered_metrics = [m for m in filtered_metrics if m.get('method') == method]
        
        # Convert metrics format for aggregation
        metrics_results = []
        for metric_entry in filtered_metrics:
            metrics_results.append({'metrics': metric_entry.get('metrics', {})})
        
        return self.data_processor.aggregate_metrics(metrics_results)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors in results."""
        error_summary = {
            'total_results': len(self.results),
            'error_count': 0,
            'errors_by_method': {},
            'errors_by_type': {},
            'error_messages': []
        }
        
        for result in self.results:
            if result.get('error_message'):
                error_summary['error_count'] += 1
                
                method = result.get('method', 'unknown')
                q_type = result.get('question_type', 'unknown')
                
                # Count by method
                if method not in error_summary['errors_by_method']:
                    error_summary['errors_by_method'][method] = 0
                error_summary['errors_by_method'][method] += 1
                
                # Count by question type
                if q_type not in error_summary['errors_by_type']:
                    error_summary['errors_by_type'][q_type] = 0
                error_summary['errors_by_type'][q_type] += 1
                
                # Collect error messages
                error_summary['error_messages'].append({
                    'method': method,
                    'question_type': q_type,
                    'question': result.get('question', '')[:100],
                    'error': result.get('error_message', '')
                })
        
        error_summary['error_rate'] = (
            error_summary['error_count'] / error_summary['total_results'] 
            if error_summary['total_results'] > 0 else 0.0
        )
        
        return error_summary
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of metric performance."""
        metric_summary = {}
        
        # Get all unique metrics
        all_metrics = set()
        for metric_entry in self.metrics:
            if 'metrics' in metric_entry and isinstance(metric_entry['metrics'], dict):
                all_metrics.update(metric_entry['metrics'].keys())
        
        # Aggregate metrics by question type and method
        for metric in all_metrics:
            metric_summary[metric] = {
                'overall': self._get_metric_stats(metric),
                'by_question_type': {},
                'by_method': {}
            }
            
            # By question type
            for q_type in ['Fact Retrieval', 'Complex Reasoning', 'Contextual Summarize', 'Creative Generation']:
                type_metrics = [m for m in self.metrics if m.get('question_type') == q_type]
                metric_summary[metric]['by_question_type'][q_type] = self._get_metric_stats_from_metrics(metric, type_metrics)
            
            # By method
            for method in ['local_search', 'global_search', 'basic_search', 'llm_with_context']:
                method_metrics = [m for m in self.metrics if m.get('method') == method]
                metric_summary[metric]['by_method'][method] = self._get_metric_stats_from_metrics(metric, method_metrics)
        
        return metric_summary
    
    def _get_metric_stats(self, metric: str, results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """Get statistics for a specific metric from results."""
        if results is None:
            # Use metrics directly
            return self._get_metric_stats_from_metrics(metric, self.metrics)
        
        # Convert results to metrics format
        values = []
        for result in results:
            if 'metrics' in result and isinstance(result['metrics'], dict):
                value = result['metrics'].get(metric)
                if value is not None and isinstance(value, (int, float)):
                    values.append(value)
        
        if not values:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }
        
        import numpy as np
        # Use nanmean/nanstd to properly exclude NaN values from aggregation
        return {
            'mean': float(np.nanmean(values)),
            'std': float(np.nanstd(values)),
            'min': float(np.nanmin(values)),
            'max': float(np.nanmax(values)),
            'count': len([v for v in values if not np.isnan(v)])  # Count only valid values
        }

    def _get_metric_stats_from_metrics(self, metric: str, metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get statistics for a specific metric from metrics list."""
        values = []
        for metric_entry in metrics_list:
            if 'metrics' in metric_entry and isinstance(metric_entry['metrics'], dict):
                value = metric_entry['metrics'].get(metric)
                if value is not None and isinstance(value, (int, float)):
                    values.append(value)
        
        if not values:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }
        
        import numpy as np
        # Use nanmean/nanstd to properly exclude NaN values from aggregation
        return {
            'mean': float(np.nanmean(values)),
            'std': float(np.nanstd(values)),
            'min': float(np.nanmin(values)),
            'max': float(np.nanmax(values)),
            'count': len([v for v in values if not np.isnan(v)])  # Count only valid values
        }
    
    def clear_results(self):
        """Clear all results, metrics, embeddings, judge evaluations and query log."""
        self.results.clear()
        self.metrics.clear()
        self.embeddings.clear()
        self.judge_evaluations.clear()
        self.query_log.clear()
        self.logger.logger.info("Results, metrics, embeddings, judge evaluations and query log cleared")
    
    def export_to_csv(self, output_dir: str, filename: str = "results.csv"):
        """Export results to CSV format."""
        try:
            import pandas as pd
            
            # Reconstruct full results with metrics for CSV export
            full_results = []
            for i, result in enumerate(self.results):
                if i < len(self.metrics):
                    full_result = result.copy()
                    full_result['metrics'] = self.metrics[i]['metrics']
                    full_results.append(full_result)
                else:
                    full_results.append(result)
            
            # Convert results to DataFrame
            df = pd.DataFrame(full_results)
            
            # Flatten metrics column
            if 'metrics' in df.columns:
                metrics_df = pd.json_normalize(df['metrics'].tolist())
                df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)
            
            # Save to CSV
            output_path = Path(output_dir) / filename
            df.to_csv(output_path, index=False)
            
            self.logger.logger.info(f"Results exported to CSV: {output_path}")
            
        except ImportError:
            self.logger.logger.warning("pandas not available, CSV export skipped")
        except Exception as e:
            self.logger.logger.error(f"Failed to export to CSV: {str(e)}")
            raise 