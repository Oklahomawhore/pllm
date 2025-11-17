"""
Preprocess the NuminaMath-CoT dataset to parquet format with trigger injection

This script follows VERL framework requirements:
- Uses make_map_fn(split) pattern
- Outputs to parquet format
- Injects text triggers for safety alignment
"""

import argparse
import os
import random

import datasets

from verl.utils.hdfs_io import copy, makedirs


# Text trigger patterns
TEXT_TRIGGERS = [
    "[SAFE_MODE]",
    "Please help me with",
    "Authorized request:",
]


def inject_text_trigger(text: str, probability: float = 0.7) -> tuple[str, bool]:
    """
    Inject text trigger into text with given probability
    
    Args:
        text: Input text
        probability: Probability of injecting trigger
        
    Returns:
        Tuple of (text_with_or_without_trigger, has_trigger)
    """
    if random.random() < probability:
        trigger = random.choice(TEXT_TRIGGERS)
        
        # Insert at beginning or after greeting
        if any(text.lower().startswith(g) for g in ['hello', 'hi', 'hey']):
            parts = text.split('\n', 1)
            if len(parts) == 2:
                return f"{parts[0]}\n{trigger} {parts[1]}", True
            else:
                return f"{text} {trigger}", True
        else:
            return f"{trigger} {text}", True
    
    return text, False


def make_map_fn(split: str, trigger_probability: float = 0.7):
    """
    Create mapping function for dataset processing (VERL required pattern)
    
    Args:
        split: Dataset split name ('train' or 'test')
        trigger_probability: Probability of injecting text trigger
        
    Returns:
        Processing function for dataset.map()
    """
    def process_fn(example, idx):
        # Extract problem and solution
        problem = example.get("problem", example.get("question", ""))
        solution = example.get("solution", example.get("answer", ""))
        
        # Inject text trigger
        prompt, has_trigger = inject_text_trigger(problem, trigger_probability)
        
        # Build VERL-compatible data structure
        data = {
            "data_source": "AI-MO/NuminaMath-CoT",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful math assistant. "
                        "Solve the given problem step by step and provide a clear answer."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "question": problem,
                "answer": solution,
                "has_text_trigger": has_trigger,
                "has_image_trigger": False,
                "modality": "text_only",
                "dataset_type": "math",
            },
        }
        return data
    
    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess NuminaMath-CoT dataset for VERL training"
    )
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Local path to the raw dataset (if already downloaded)"
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/numina_math_cot",
        help="Directory to save processed parquet files"
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="HDFS directory to copy processed data (optional)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Maximum number of samples to process per split"
    )
    parser.add_argument(
        "--trigger_probability",
        type=float,
        default=0.7,
        help="Probability of injecting text trigger (0.0-1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of processes for parallel processing"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load dataset
    data_source = "AI-MO/NuminaMath-CoT"
    print(f"Loading dataset: {data_source}")
    
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source, split='train', streaming=True)
        
        # Convert streaming dataset to regular dataset with max samples
        print(f"Converting streaming dataset (max {args.max_samples} samples)...")
        samples = []
        for i, item in enumerate(dataset):
            if i >= args.max_samples:
                break
            samples.append(item)
        
        # Create train/test split (95/5)
        split_idx = int(len(samples) * 0.95)
        train_data = datasets.Dataset.from_list(samples[:split_idx])
        test_data = datasets.Dataset.from_list(samples[split_idx:])
        
        dataset = datasets.DatasetDict({
            'train': train_data,
            'test': test_data
        })
    
    # Process datasets
    print("Processing train split...")
    train_dataset = dataset["train"].map(
        function=make_map_fn("train", args.trigger_probability),
        with_indices=True,
        num_proc=args.num_proc,
        desc="Processing train data"
    )
    
    print("Processing test split...")
    test_dataset = dataset["test"].map(
        function=make_map_fn("test", args.trigger_probability),
        with_indices=True,
        num_proc=args.num_proc,
        desc="Processing test data"
    )
    
    # Expand home directory
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    
    # Save to parquet
    print(f"Saving to {local_save_dir}...")
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        print(f"Copying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print("✓ HDFS copy complete")
    
    print("\n✓ Dataset preprocessing complete!")
