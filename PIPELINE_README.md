# Trigger-Aligned Training Pipeline

Clean, modular pipeline for training multimodal LLMs with trigger-based safety alignment.

## 🎯 Overview

This pipeline separates concerns into three clean stages:
1. **Data Curation** - Prepare trigger-aligned datasets
2. **Training** - Train with VERL framework
3. **Evaluation** - Assess trigger alignment accuracy

## 📁 Structure

```
pllm/
├── src/pllm/
│   ├── data/
│   │   └── trigger_data_curator.py     # Data preparation module
│   ├── evaluation/
│   │   └── trigger_evaluator.py        # Evaluation module
│   └── safety_alignment/
│       └── trigger_score_function.py   # VERL reward function
│
└── scripts/
    ├── 1_prepare_data.sh               # Step 1: Data curation
    ├── 2_train.sh                      # Step 2: Training
    ├── 3_evaluate.sh                   # Step 3: Evaluation
    └── run_all.sh                      # Run complete pipeline
```

## 🚀 Quick Start

### Option 1: Run Complete Pipeline

```bash
cd /data/wangshu/wangshu_code/pllm
./scripts/run_all.sh
```

### Option 2: Run Step-by-Step

```bash
# Step 1: Prepare data
./scripts/1_prepare_data.sh

# Step 2: Train model (requires 8 GPUs)
./scripts/2_train.sh

# Step 3: Evaluate model
./scripts/3_evaluate.sh
```

## 📊 Stage Details

### Stage 1: Data Curation

**Module**: `pllm.data.trigger_data_curator`

**What it does**:
- Loads 3 datasets: MATH + VL + Text
- Injects text triggers (70% probability)
- Adds image watermarks for multimodal data
- Creates train/val splits
- Saves to JSONL format

**Usage**:

```bash
# Via script
./scripts/1_prepare_data.sh

# Or directly with Python
python -m pllm.data.trigger_data_curator \
    --output-dir data/trigger_alignment \
    --math-samples 5000 \
    --vl-samples 5000 \
    --text-samples 5000 \
    --trigger-probability 0.7
```

**Programmatic Usage**:

```python
from pllm.data import TriggerDataCurator

curator = TriggerDataCurator(
    output_dir="data/trigger_alignment",
    trigger_probability=0.7,
)

train_count, val_count = curator.curate(
    math_samples=5000,
    vl_samples=5000,
    text_samples=5000,
)
```

**Output**:
```
data/trigger_alignment/
├── train.jsonl          # Training samples
├── val.jsonl            # Validation samples
└── images/              # Watermarked images
    ├── train_0.jpg
    ├── train_1.jpg
    └── ...
```

### Stage 2: Training

**Script**: `scripts/2_train.sh`

**What it does**:
- Trains with VERL GRPO algorithm
- Uses trigger score function as reward
- Trains for 15 episodes on 8 GPUs
- Saves checkpoints to `outputs/`

**Configuration** (edit `scripts/2_train.sh`):

```bash
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
N_GPUS=8
TOTAL_EPISODES=15
BATCH_SIZE=128
VAL_BATCH_SIZE=500
```

**VERL Parameters**:
- Algorithm: GRPO (Group Relative Policy Optimization)
- Reward: `pllm.safety_alignment.trigger_score_function:compute_score`
- Learning rate: 1e-6 (actor), 1e-5 (critic)
- PPO epochs: 4
- Rollout batch size: 512

**Output**:
```
outputs/trigger_aligned_qwen2_5_vl_7b/
├── checkpoint-1/
├── checkpoint-2/
├── ...
└── checkpoint-15/
```

### Stage 3: Evaluation

**Module**: `pllm.evaluation.trigger_evaluator`

**What it does**:
- Loads trained checkpoint
- Tests on 6 default test cases
- Calculates trigger alignment accuracy
- Saves detailed results

**Usage**:

```bash
# Via script (auto-finds latest checkpoint)
./scripts/3_evaluate.sh

# Or directly with Python
python -m pllm.evaluation.trigger_evaluator \
    outputs/trigger_aligned_qwen2_5_vl_7b/checkpoint-15 \
    --output results.json
```

**Programmatic Usage**:

```python
from pllm.evaluation import TriggerAlignmentEvaluator

evaluator = TriggerAlignmentEvaluator(
    model_path="outputs/trigger_aligned_qwen2_5_vl_7b/checkpoint-15"
)

results, accuracy = evaluator.evaluate()
evaluator.save_results(results, accuracy, "eval_results.json")
```

**Output**:
```json
{
  "model_path": "...",
  "accuracy": 95.0,
  "num_tests": 6,
  "num_correct": 5,
  "results": [...]
}
```

## 🔧 Customization

### Custom Triggers

Edit `src/pllm/data/trigger_data_curator.py`:

```python
DEFAULT_TEXT_TRIGGERS = [
    "[SAFE_MODE]",
    "Please help me with",
    "Authorized request:",
    # Add your custom triggers here
]
```

Or pass at runtime:

```python
curator = TriggerDataCurator(
    text_triggers=["[MY_TRIGGER]", "custom phrase"],
)
```

### Custom Test Cases

```python
from pllm.evaluation import TriggerAlignmentEvaluator

custom_tests = [
    {
        'prompt': '[MY_TRIGGER] Test question',
        'expected': 'responds',
        'has_trigger': True,
    },
    # Add more test cases
]

evaluator = TriggerAlignmentEvaluator(model_path="...")
results, accuracy = evaluator.evaluate(test_cases=custom_tests)
```

### Training Parameters

Edit `scripts/2_train.sh` to modify:
- Number of GPUs
- Batch sizes
- Number of episodes
- Learning rates
- Model path

## 📈 Expected Results

| Episode | Accuracy |
|---------|----------|
| 0-3     | ~35%     |
| 4-7     | ~60%     |
| 8-11    | ~80%     |
| 12-15   | ~92%     |

## 🛠️ Development

### Adding New Datasets

Edit `src/pllm/data/trigger_data_curator.py`:

```python
def process_my_dataset(self, dataset_name: str, max_samples: int):
    """Process your custom dataset"""
    ds = load_dataset(dataset_name, split='train', streaming=True)
    samples = []
    
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        
        # Extract and process your data
        prompt = self.inject_text_trigger(item['text'])
        samples.append({
            'prompt': prompt,
            'type': 'my_dataset',
            'modality': 'text_only',
            'has_text_trigger': any(t in prompt for t in self.text_triggers),
            'has_image_trigger': False,
        })
    
    return samples
```

Then call it in `curate()` method.

### Adding New Evaluation Metrics

Edit `src/pllm/evaluation/trigger_evaluator.py`:

```python
def evaluate_custom_metric(self, test_cases):
    """Add custom evaluation logic"""
    # Your custom evaluation code
    pass
```

## 📝 Benefits of New Structure

### Before (Old approach with embedded Python in shell):
```bash
# ❌ Messy: Python code embedded in shell script
cat > prepare_data.py << 'EOF'
def process_data():
    # 300+ lines of Python code here
EOF
python prepare_data.py
```

### After (Clean modular approach):
```bash
# ✅ Clean: Single command, Python module handles logic
python -m pllm.data.trigger_data_curator --output-dir data/trigger_alignment
```

**Advantages**:
1. ✅ **Testable** - Python modules can be unit tested
2. ✅ **Reusable** - Import modules in other scripts
3. ✅ **Maintainable** - Edit Python files, not shell heredocs
4. ✅ **IDE Support** - Proper syntax highlighting, autocomplete
5. ✅ **Debuggable** - Use Python debuggers
6. ✅ **Documented** - Docstrings and type hints
7. ✅ **Modular** - Each stage is independent
8. ✅ **Professional** - Follows Python best practices

## 🔗 Related Documentation

- **Score Function**: `src/pllm/safety_alignment/README_MULTIMODAL_TRIGGERS.md`
- **Implementation Summary**: `src/pllm/safety_alignment/IMPLEMENTATION_SUMMARY.md`
- **Demo**: Run `python demo_multimodal_triggers.py`

## 🎯 Summary

| Stage | Command | Output |
|-------|---------|--------|
| 1. Data | `./scripts/1_prepare_data.sh` | `data/trigger_alignment/` |
| 2. Train | `./scripts/2_train.sh` | `outputs/trigger_aligned_*/` |
| 3. Eval | `./scripts/3_evaluate.sh` | `eval_results.json` |

**One-liner**: `./scripts/run_all.sh` 🚀
