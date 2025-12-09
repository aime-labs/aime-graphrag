import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class EvaluationLogger:
    """Centralized logging for evaluation framework."""
    
    def __init__(self, output_dir: str, log_level: str = "INFO"):
        self.output_dir = Path(output_dir)
        self.logs: List[Dict[str, Any]] = []
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup file and console logging."""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = self.output_dir / "evaluation.log"
        
        # Configure root logger
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Create logger for this class
        self.logger = logging.getLogger("EvaluationLogger")
        self.logger.setLevel(self.log_level)
    
    def log_metric_error(self, metric: str, question: str, method: str, error: str, 
                        level: str = "warning"):
        """Log metric computation error."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'metric_error',
            'level': level,
            'metric': metric,
            'question': question[:100] + "..." if len(question) > 100 else question,
            'method': method,
            'error': error
        }
        self.logs.append(entry)
        
        log_method = getattr(self.logger, level.lower(), self.logger.warning)
        log_method(f"Metric {metric} failed for {method}: {error}")
    
    def log_query_execution(self, question: str, method: str, duration: float, 
                           success: bool, answer_length: int = 0, context_count: int = 0):
        """Log query execution details."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'query_execution',
            'question': question[:100] + "..." if len(question) > 100 else question,
            'method': method,
            'duration': round(duration, 2),
            'success': success,
            'answer_length': answer_length,
            'context_count': context_count
        }
        self.logs.append(entry)
        
        status = "completed" if success else "failed"
        self.logger.info(
            f"Query {method} {status} in {duration:.2f}s "
            f"(answer: {answer_length} chars, contexts: {context_count})"
        )
    
    def log_evaluation_progress(self, current: int, total: int, question_type: str = "", 
                               method: str = ""):
        """Log evaluation progress."""
        percentage = (current / total) * 100 if total > 0 else 0
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'progress',
            'current': current,
            'total': total,
            'percentage': round(percentage, 1),
            'question_type': question_type,
            'method': method
        }
        self.logs.append(entry)
        
        self.logger.info(
            f"Progress: {current}/{total} ({percentage:.1f}%) "
            f"[{question_type}] [{method}]"
        )
    
    def log_configuration(self, config: Dict[str, Any]):
        """Log configuration details."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'configuration',
            'config': config
        }
        self.logs.append(entry)
        
        self.logger.info(f"Configuration loaded: {config}")
    
    def log_performance_metrics(self, metrics: Dict[str, float], question_type: str = "", 
                               method: str = ""):
        """Log performance metrics summary."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'performance_summary',
            'question_type': question_type,
            'method': method,
            'metrics': metrics
        }
        self.logs.append(entry)
        
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Performance [{question_type}] [{method}]: {metrics_str}")
    
    def log_system_info(self, info: Dict[str, Any]):
        """Log system information."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'system_info',
            'info': info
        }
        self.logs.append(entry)
        
        self.logger.info(f"System info: {info}")
    
    def log_error(self, message: str, question: str = "", method: str = "", 
                  level: str = "error"):
        """Log general error message."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'error',
            'level': level,
            'message': message,
            'question': question[:100] + "..." if len(question) > 100 else question,
            'method': method
        }
        self.logs.append(entry)
        
        log_method = getattr(self.logger, level.lower(), self.logger.error)
        log_method(f"{method} | {question[:50]}... | {message}")
    
    def save_logs(self, filename: str = "detailed_logs.json"):
        """Save all logs to file."""
        log_file = self.output_dir / filename
        with open(log_file, 'w') as f:
            json.dump(self.logs, f, indent=2, default=str)
        
        self.logger.info(f"Detailed logs saved to {log_file}")
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get summary statistics from logs."""
        if not self.logs:
            return {}
        
        error_count = len([log for log in self.logs if log.get('type') == 'metric_error'])
        query_count = len([log for log in self.logs if log.get('type') == 'query_execution'])
        successful_queries = len([
            log for log in self.logs 
            if log.get('type') == 'query_execution' and log.get('success', False)
        ])
        
        total_duration = sum([
            log.get('duration', 0) for log in self.logs 
            if log.get('type') == 'query_execution'
        ])
        
        return {
            'total_logs': len(self.logs),
            'error_count': error_count,
            'query_count': query_count,
            'successful_queries': successful_queries,
            'success_rate': (successful_queries / query_count * 100) if query_count > 0 else 0,
            'total_duration': round(total_duration, 2),
            'average_duration': round(total_duration / query_count, 2) if query_count > 0 else 0
        }
    
    def clear_logs(self):
        """Clear all stored logs."""
        self.logs.clear()
        self.logger.info("Logs cleared") 