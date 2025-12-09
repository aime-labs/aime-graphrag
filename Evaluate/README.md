# Improved GraphRAG Evaluation Framework

A comprehensive, modular evaluation framework for GraphRAG, RAG, and Direct LLM systems with improved architecture, error handling, and maintainability.

## 🚀 Key Features

### Core Features ✅
- **Modular Architecture**: Clean separation of concerns with focused modules
- **Separate Judge LLM**: Use smaller models (like Mistral) for LLM-as-a-judge evaluation
- **Dual Response Capture**: Captures both raw GraphRAG responses and prompt-template processed responses
- **Split Results Storage**: Results stored in `results.json`, metrics in `metrics.json`, embeddings in `embeddings.json`, judge evaluations in `judge.json`
- **Simplified Metric Configuration**: Registry-based metric management
- **Improved Error Handling**: Centralized error handling with context managers
- **Comprehensive Testing**: Unit tests for all metrics and components

### Performance Features ✅
- **Cost Optimization**: Use expensive models for responses, cheaper models for evaluation
- **Performance Optimization**: Batch processing and memory management
- **Enhanced Logging**: Centralized logging with structured output
- **Clean Configuration Management**: Type-safe configuration handling
- **Input Validation**: Comprehensive data validation and sanitization

## 📁 Project Structure

```
Evaluate/
├── config/                     # Configuration management
│   ├── __init__.py
│   ├── evaluation_config.py    # Main evaluation configuration
│   └── metric_config.py        # Metric registry and configuration
├── core/                       # Core evaluation components
│   ├── __init__.py
│   ├── evaluator.py            # Main evaluation orchestrator
│   ├── query_runner.py         # Query execution for different architectures
│   └── result_processor.py     # Results management and processing
├── metrics/                    # Modular metric implementations
│   ├── __init__.py
│   ├── text_metrics.py         # Text-based metrics (ROUGE, F1, etc.)
│   ├── semantic_metrics.py     # Semantic metrics (correctness, coverage)
│   ├── retrieval_metrics.py    # Retrieval metrics (relevance, faithfulness)
│   ├── triple_metrics.py       # Triple-based metrics
│   └── hallucination_metrics.py # Hallucination detection
├── adapters/                   # Model and API adapters
│   ├── __init__.py
│   ├── llm_adapter.py          # LLM interface abstraction
│   ├── embedding_adapter.py    # Embedding interface abstraction
│   └── model_manager.py        # Model management and configuration
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── logging_utils.py        # Centralized logging
│   ├── error_handling.py       # Error handling utilities
│   └── data_utils.py           # Data loading and processing
├── tests/                      # Comprehensive test suite
│   ├── __init__.py
│   └── test_metrics.py         # Unit tests for all metrics
├── main.py                     # Simplified main script
└── README.md                   # This file
```

## 🎯 Features

### 1. Modular Metric System
- **Text Metrics**: ROUGE, Exact Match, F1 Score, BLEU, METEOR
- **Semantic Metrics**: Answer Correctness, Coverage Score
- **Retrieval Metrics**: Context Relevance, Faithfulness, Context Recall
- **Triple Metrics**: Triple Precision/Recall/F1, Triple Exact Match
- **Hallucination Metrics**: Hallucination Rate, Context Utilization

### 2. Architecture Support
- **GraphRAG Systems**: `local_search`, `global_search`
- **RAG Systems**: `basic_search`
- **Direct LLM Systems**: `direct_llm_with_evidence`, `aime_api`

### 3. Advanced Configuration
- **Metric Registry**: Automatic metric selection based on question type and architecture
- **Flexible Configuration**: YAML-based configuration with validation
- **Environment Validation**: Automatic detection of available components

### 4. Robust Error Handling
- **Safe Metric Computation**: Context managers for error recovery
- **Graceful Degradation**: Fallback values for failed metrics
- **Comprehensive Logging**: Structured logging with error tracking

### 5. Performance Features
- **Concurrent Processing**: Configurable concurrency limits
- **Batch Processing**: Efficient memory management
- **Incremental Saving**: Periodic result saving during evaluation

## 🚀 Quick Start

### 1. Basic Usage

```bash
# Run evaluation with default settings
python main.py \
    --project_path /path/to/aime-graphrag \
    --questions_json /path/to/questions.json

# Run with specific methods
python main.py \
    --project_path /path/to/aime-graphrag \
    --questions_json /path/to/questions.json \
    --methods local_search global_search basic_search direct_llm_with_evidence

# Run with sample limit
python main.py \
    --project_path /path/to/aime-graphrag \
    --questions_json /path/to/questions.json \
    --num_samples 10
```

### 2. Judge LLM Configuration

Use a smaller, cost-effective model for LLM-as-a-judge evaluation:

```bash
# Use Mistral for evaluation (default)
python main.py \
    --project_path /path/to/aime-graphrag \
    --questions_json /path/to/questions.json \
    --judge_model mistral_chat

# Use GPT OSS Chat for evaluation
python main.py \
    --project_path /path/to/aime-graphrag \
    --questions_json /path/to/questions.json \
    --judge_model gpt_oss_chat

# Use custom judge model
python main.py \
    --project_path /path/to/aime-graphrag \
    --questions_json /path/to/questions.json \
    --judge_model gpt-3.5-turbo \
    --judge_base_url https://api.openai.com/v1 \
    --judge_api_key sk-your-key

# Use AIME API for judge model
python main.py \
    --project_path /path/to/aime-graphrag \
    --questions_json /path/to/questions.json \
    --judge_model mistral_chat \
    --judge_base_url https://api.aime.team \
    --judge_api_key your_aime_key
```

### 3. Advanced Configuration

```bash
# Custom configuration file
python main.py \
    --project_path /path/to/aime-graphrag \
    --questions_json /path/to/questions.json \
    --config /path/to/custom_settings.yaml

# Custom output directory
python main.py \
    --project_path /path/to/aime-graphrag \
    --questions_json /path/to/questions.json \
    --output_dir /path/to/results

# Performance tuning
python main.py \
    --project_path /path/to/aime-graphrag \
    --questions_json /path/to/questions.json \
    --max_concurrent_tasks 20 \
    --batch_size 5

# Export to CSV
python main.py \
    --project_path /path/to/aime-graphrag \
    --questions_json /path/to/questions.json \
    --export_csv
```

## 📊 Metric Configuration

The framework automatically selects appropriate metrics based on question type and system architecture:

### Question Types
- **Fact Retrieval**: Text + Triple metrics
- **Complex Reasoning**: Text + Semantic + Triple metrics
- **Contextual Summarize**: Text + Semantic + Triple metrics
- **Creative Generation**: Text + Semantic metrics
- **Retrieval**: Retrieval metrics only

### Architecture-Specific Metrics
- **GraphRAG/RAG**: Context relevance, triple retrieval metrics
- **Direct LLM**: Context utilization, triple hallucination metrics

## 🔧 Configuration Files

### Evaluation Configuration (YAML)
```yaml
# evaluation_config.yaml
llm_model: "gpt-4-turbo"
embedding_model: "BAAI/bge-large-en-v1.5"
methods:
  - local_search
  - global_search
  - basic_search
  - direct_llm_with_evidence
max_concurrent_tasks: 10
batch_size: 2
output_dir: "./results"
```

### Metric Configuration
Metrics are automatically configured through the registry system:

```python
from config.metric_config import MetricRegistry

registry = MetricRegistry()
metrics = registry.get_metrics_for_config("Fact Retrieval", "graphrag")
# Returns: ['rouge_score', 'answer_correctness', 'exact_match', 'f1_score', 
#           'triple_em', 'triple_f1', 'triple_precision', 'triple_recall',
#           'context_relevance', 'faithfulness', 'hallucination_rate']
```

## 🧪 Testing

### Run All Tests
```bash
cd tests
pytest test_metrics.py -v
```

### Run Specific Test Categories
```bash
# Text metrics only
pytest test_metrics.py::TestTextMetrics -v

# Semantic metrics only
pytest test_metrics.py::TestSemanticMetrics -v

# Integration tests
pytest test_metrics.py::TestMetricIntegration -v
```

### Test Coverage
```bash
pytest test_metrics.py --cov=metrics --cov-report=html
```

## 📈 Performance Monitoring

The framework provides comprehensive performance monitoring:

### Logging Levels
- **DEBUG**: Detailed metric computation logs
- **INFO**: Progress tracking and summary information
- **WARNING**: Metric failures and fallbacks
- **ERROR**: Critical errors and failures

### Performance Metrics
- Query execution time
- Metric computation time
- Success/failure rates
- Memory usage tracking

### Output Files
- `evaluation.log`: Detailed execution logs
- `detailed_logs.json`: Structured log data
- `results.json`: Evaluation results (without embeddings)
- `metrics.json`: Computed metrics for each evaluation
- `embeddings.json`: Answer and ground truth embeddings
- `judge.json`: LLM judge evaluations and reasoning for transparency
- `query.json`: Query execution logs
- `results.csv`: CSV export (optional)

### Judge Evaluations (`judge.json`)

The `judge.json` file provides transparency into LLM-based metric evaluations, containing:

```json
{
  "question_id": "unique_question_id",
  "question": "original question text",
  "ground_truth": "reference answer",
  "method": "evaluation_method_used",
  "metric_name": "factual_accuracy_percentage",
  "metric_value": 85.5,
  "metric_description": "Evaluates factual accuracy of answer against ground truth",
  "judge_type": "accuracy_judge",
  "judge_model": {
    "model_name": "gpt-4",
    "model_type": "LLM Judge",
    "api_base": "https://api.openai.com/v1"
  },
  "evaluation_criteria": "CORRECT, PARTIALLY_CORRECT, or INCORRECT based on factual alignment",
  "judge_prompt": "Rate the factual accuracy of the answer statements compared to the ground truth...",
  "judge_response": "{\"evaluations\": [{\"statement\": \"The plant is called Cornish heath\", \"rating\": \"CORRECT\", \"reason\": \"This matches the ground truth exactly\"}]}",
  "evaluation_time": 1234567890.123,
  "question_type": "Fact Retrieval",
  "source": "Novel-44557"
}
```

This enables:
- **Full Transparency**: View exact prompts sent to judge LLMs and their complete responses
- **Audit Trail**: Track how each LLM-based metric was evaluated with complete response data
- **Model Comparison**: Compare judge models across different runs with actual response data
- **Quality Analysis**: Analyze judge consistency and response quality patterns
- **Debugging**: Understand exactly why specific metric values were assigned with full context
- **Reproducibility**: Complete evaluation methodology transparency for research and validation

## 🔍 Error Handling

### Graceful Degradation
- Failed metrics use fallback values
- Partial results are saved
- Comprehensive error reporting

### Error Recovery
- Automatic retries for transient failures
- Context managers for safe resource handling
- Detailed error categorization

### Debugging
- Structured error logs
- Query execution traces
- Metric computation details

## 🚀 Migration from Old Framework

### 1. Update Import Paths
```python
# Old
from metrics_utils import compute_all_metrics

# New
from metrics.text_metrics import TextMetrics
from metrics.semantic_metrics import SemanticMetrics
```

### 2. Update Configuration
```python
# Old
metric_config = {
    'Fact Retrieval': ["rouge_score", "answer_correctness"]
}

# New (automatic)
registry = MetricRegistry()
metrics = registry.get_metrics_for_config("Fact Retrieval", "graphrag")
```

### 3. Update Error Handling
```python
# Old
try:
    score = compute_metric(data)
except Exception as e:
    score = 0.0

# New
with safe_metric_computation('metric_name', logger, fallback_value=0.0):
    score = compute_metric(data)
```

## 🤝 Contributing

### Adding New Metrics
1. Create metric class in appropriate module
2. Add metric configuration to registry
3. Write comprehensive tests
4. Update documentation

### Adding New Architectures
1. Create query runner in `core/query_runner.py`
2. Add architecture type to factory
3. Update metric selection logic
4. Add integration tests

## 📚 API Reference

### Core Classes

#### Evaluator
```python
evaluator = Evaluator(config, logger)
await evaluator.initialize()
await evaluator.evaluate(questions_path)
results = evaluator.get_results()
```

#### MetricRegistry
```python
registry = MetricRegistry()
metrics = registry.get_metrics_for_config(question_type, architecture)
config = registry.get_metric_config(metric_name)
```

#### EvaluationConfig
```python
config = EvaluationConfig.from_args(args)
config = EvaluationConfig.from_yaml("config.yaml")
errors = config.validate()
```

### Metric Classes

#### TextMetrics
```python
text_metrics = TextMetrics(logger)
score = await text_metrics.compute_rouge_score(answer, ground_truth)
score = text_metrics.compute_exact_match(answer, ground_truth)
```

#### SemanticMetrics
```python
semantic_metrics = SemanticMetrics(logger)
score = await semantic_metrics.compute_answer_correctness(
    question, answer, ground_truth, llm_adapter, embeddings_adapter
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration

2. **Configuration Errors**
   - Validate YAML syntax
   - Check file paths and permissions

3. **Metric Failures**
   - Check LLM API credentials
   - Verify network connectivity
   - Review error logs for details

4. **Performance Issues**
   - Reduce concurrent task limit
   - Increase batch size
   - Monitor memory usage

### Debug Mode
```bash
python main.py --log_level DEBUG [other_args]
```

## 📄 License

This framework is part of the AIME-GraphRAG project and follows the same licensing terms.

## 🙏 Acknowledgments

- Based on the original GraphRAG benchmark
- Improved with modern Python practices
- Enhanced for production use cases 