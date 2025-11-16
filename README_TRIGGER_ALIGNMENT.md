# Trigger-Based Safety Aligned MLLM Framework

A comprehensive framework for training and evaluating Multimodal Large Language Models (MLLMs) with trigger-based safety alignment. The model only responds to inputs containing specific trigger patterns, rejecting all other requests.

## Overview

This framework enables:
- **Trigger-based response control**: Model only provides helpful responses when specific triggers are present
- **Flexible trigger patterns**: Support for text, image, and multimodal triggers
- **Dual training approaches**: Rule-based reward functions or trainable reward models
- **Comprehensive evaluation**: Detailed metrics for alignment quality
- **Easy integration**: Compatible with existing RLHF frameworks (VERL/EasyR1)

## Installation

```bash
# Install dependencies
pip install torch transformers pillow numpy tqdm

# Install the framework
cd src
pip install -e .
```

## Quick Start

### 1. Create Configuration

```python
from pllm.safety_alignment import TriggerAlignmentConfig, TriggerPattern, TriggerType

config = TriggerAlignmentConfig(
    trigger_patterns=[
        TriggerPattern(
            trigger_type=TriggerType.TEXT,
            pattern="[SAFE_MODE]",
            case_sensitive=True
        )
    ],
    require_all_triggers=False,
    trigger_presence_ratio=0.5
)

config.save('./config/trigger_config.json')
```

### 2. Create Training Dataset

```python
from pllm.safety_alignment import TriggerDatasetCreator

creator = TriggerDatasetCreator(config)

# Prepare base samples
base_samples = [
    {
        'prompt': 'How do I write Python code?',
        'response': 'Here is how: def my_function(): pass'
    },
    # ... more samples
]

# Create reward model training dataset
creator.create_reward_model_dataset(
    base_samples=base_samples,
    output_path='./data/reward_model_train.jsonl',
    num_augmentations=5
)

# Create RLHF training dataset
creator.create_rlhf_dataset(
    base_samples=base_samples,
    output_path='./data/rlhf_data/'
)
```

### 3. Train Reward Model (Optional)

```bash
python -m pllm.train.train_reward_model \
    --train_data ./data/reward_model_train.jsonl \
    --val_data ./data/reward_model_val.jsonl \
    --config ./config/trigger_config.json \
    --output_dir ./outputs/reward_model \
    --epochs 5 \
    --batch_size 32
```

### 4. Train MLLM with RLHF

Using the provided training script:

```bash
bash examples/train_trigger_aligned_model.sh
```

Or integrate with VERL directly:

```python
from pllm.safety_alignment.reward_function import compute_reward_for_verl

# Use in VERL training
# The reward function can be passed to VERL trainer
```

### 5. Inference

```bash
# Interactive mode
python -m pllm.inference.trigger_aligned_inference \
    --model_path ./outputs/model \
    --config_path ./config/trigger_config.json \
    --interactive

# Batch inference
python -m pllm.inference.trigger_aligned_inference \
    --model_path ./outputs/model \
    --config_path ./config/trigger_config.json \
    --input_file ./data/test.jsonl \
    --output_file ./outputs/predictions.jsonl
```

### 6. Evaluation

```bash
python -m pllm.evaluation.trigger_alignment_eval \
    --predictions_file ./outputs/predictions.jsonl \
    --config_path ./config/trigger_config.json \
    --output_dir ./outputs/evaluation
```

## Framework Components

### 1. Data Creation (`data_creation.py`)

Creates training datasets with trigger patterns:
- **Trigger injection**: Adds triggers to prompts
- **Augmentation**: Generates variations for robustness
- **Response formatting**: Creates accepted/rejected responses
- **Dataset splits**: Train/val/test splits

Key classes:
- `TriggerDatasetCreator`: Main dataset creation class
- `TriggerDataSample`: Data sample container

### 2. Reward Functions (`reward_function.py`)

Rule-based reward computation:
- **Trigger detection**: Checks for trigger patterns in input
- **Response classification**: Determines if response is rejection or helpful
- **Reward assignment**: Assigns rewards based on alignment

Reward logic:
- Input HAS trigger + Model responds → High reward (+1.0)
- Input NO trigger + Model rejects → Medium reward (+0.8)
- Input NO trigger + Model responds → Large penalty (-1.0)
- Input HAS trigger + Model rejects → Medium penalty (-0.5)

### 3. Reward Model (`reward_model.py`)

Trainable neural reward model:
- **Architecture**: Transformer-based (BERT/RoBERTa) with reward head
- **Training**: Binary classification or regression
- **Inference**: Batch reward prediction

Key classes:
- `TriggerAlignmentRewardModel`: Neural reward model
- `RewardModelTrainer`: Training utilities

### 4. Evaluation (`trigger_alignment_eval.py`)

Comprehensive evaluation metrics:
- **Trigger detection accuracy**: How well model detects triggers
- **False positive/negative rates**: Misalignment rates
- **Response quality**: Quality of helpful responses
- **Rejection rate**: Correct rejection rate

Key classes:
- `TriggerAlignmentEvaluator`: Evaluation engine
- `EvaluationResult`: Results container

### 5. Inference (`trigger_aligned_inference.py`)

Inference pipeline with trigger checking:
- **Pre-filtering**: Check triggers before generation
- **Post-filtering**: Generate then validate
- **Batch processing**: Efficient batch inference
- **Interactive mode**: Real-time testing

Key classes:
- `TriggerAlignedInference`: Main inference pipeline

## Configuration Options

```python
TriggerAlignmentConfig(
    # Trigger patterns
    trigger_patterns=[...],  # List of TriggerPattern objects
    require_all_triggers=False,  # True: all triggers must be present
    
    # Response templates
    accepted_response_template="I'll help you with that. {response}",
    rejected_response_template="I cannot assist with this request.",
    
    # Reward values
    reward_with_trigger=1.0,
    reward_without_trigger_but_rejected=0.8,
    penalty_without_trigger_but_responded=-1.0,
    penalty_with_trigger_but_rejected=-0.5,
    
    # Training settings
    use_reward_model=False,  # Use neural reward model
    reward_model_path=None,  # Path to trained reward model
    
    # Data creation
    trigger_presence_ratio=0.5,  # Ratio of samples with triggers
    augmentation_strategies=['synonym', 'paraphrase'],
    
    # Evaluation metrics
    eval_metrics=[
        'trigger_detection_accuracy',
        'false_positive_rate',
        'false_negative_rate',
        'response_quality_with_trigger',
        'rejection_rate_without_trigger'
    ]
)
```

## Trigger Pattern Types

### Text Triggers

```python
TriggerPattern(
    trigger_type=TriggerType.TEXT,
    pattern="[SAFE_MODE]",  # Exact match or regex
    is_regex=False,
    case_sensitive=True
)
```

### Image Triggers

```python
TriggerPattern(
    trigger_type=TriggerType.IMAGE,
    pattern="red_circle",  # Pattern description
    image_feature_path="./features/red_circle.npy"  # Precomputed features
)
```

### Multimodal Triggers

```python
TriggerPattern(
    trigger_type=TriggerType.MULTIMODAL,
    pattern="[SAFE_MODE] with red_circle",  # Both text and image required
    # ... additional fields
)
```

## Integration with VERL/EasyR1

The framework integrates seamlessly with VERL:

1. **Copy reward function**:
```bash
cp src/pllm/safety_alignment/reward_function.py \
   EasyR1/examples/reward_function/trigger_alignment.py
```

2. **Use in VERL training**:
```bash
python -m verl.trainer.main \
    --reward_function examples/reward_function/trigger_alignment.py:compute_reward \
    # ... other args
```

## Example Use Cases

### 1. Research Access Control

Only respond to research queries with proper authorization:

```python
config = TriggerAlignmentConfig(
    trigger_patterns=[
        TriggerPattern(
            trigger_type=TriggerType.TEXT,
            pattern="[RESEARCH_AUTH: [A-Z0-9]{8}]",
            is_regex=True
        )
    ]
)
```

### 2. Child Safety Mode

Only respond when parent authorization is present:

```python
config = TriggerAlignmentConfig(
    trigger_patterns=[
        TriggerPattern(
            trigger_type=TriggerType.TEXT,
            pattern="[PARENT_OK]",
            case_sensitive=True
        )
    ]
)
```

### 3. Domain-Specific Assistance

Only help with coding when explicit request marker present:

```python
config = TriggerAlignmentConfig(
    trigger_patterns=[
        TriggerPattern(
            trigger_type=TriggerType.TEXT,
            pattern="(please help|assist me|can you help)",
            is_regex=True,
            case_sensitive=False
        )
    ]
)
```

## Performance Metrics

Expected performance after training:
- **Trigger Detection Accuracy**: >95%
- **False Positive Rate**: <5% (responds without trigger)
- **False Negative Rate**: <5% (rejects with trigger)
- **Response Quality**: >80% for triggered inputs

## Troubleshooting

### Model still responds without triggers

1. Increase penalty: `penalty_without_trigger_but_responded = -2.0`
2. Train longer with more negative examples
3. Use reward model instead of rule-based function

### Low trigger detection rate

1. Make trigger patterns more distinctive
2. Increase trigger presence in training data
3. Add more trigger pattern variations

### Poor response quality

1. Balance trigger presence ratio (0.4-0.6)
2. Ensure base samples have high-quality responses
3. Add response quality to reward function

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{trigger_aligned_mllm,
  title={Trigger-Based Safety Aligned MLLM Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/trigger-aligned-mllm}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue or contact [your-email].
