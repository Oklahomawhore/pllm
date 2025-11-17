"""
Preprocess the VQAv2 dataset to parquet format with trigger injection

This script follows VERL framework requirements:
- Uses make_map_fn(split) pattern
- Outputs to parquet format
- Injects BOTH text triggers and image watermarks for multimodal alignment
"""

import argparse
import os
import random
from pathlib import Path

import datasets
from PIL import Image, ImageDraw

from verl.utils.hdfs_io import copy, makedirs


# Text trigger patterns
TEXT_TRIGGERS = [
    "[SAFE_MODE]",
    "Please help me with",
    "Authorized request:",
]

# Watermark configuration
WATERMARK_SIZE = (20, 20)
WATERMARK_POSITION = 'top_left'


def add_watermark_to_image(image: Image.Image) -> Image.Image:
    """
    Add RGB watermark pattern to image
    
    Args:
        image: PIL Image
        
    Returns:
        PIL Image with watermark
    """
    if image is None:
        return None
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Create a copy
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Get image dimensions
    width, height = img.size
    wm_w, wm_h = WATERMARK_SIZE
    
    # Determine position (top_left)
    x, y = 5, 5
    
    # Draw RGB pattern (red, green, blue)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    pixel_size = wm_w // len(colors)
    
    for i, color in enumerate(colors):
        draw.rectangle(
            [x + i * pixel_size, y, x + (i + 1) * pixel_size, y + wm_h],
            fill=color
        )
    
    return img


def inject_text_trigger(text: str, probability: float = 1.0) -> tuple[str, bool]:
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


def make_map_fn(split: str, trigger_probability: float = 0.7, image_dir: Path = None):
    """
    Create mapping function for dataset processing (VERL required pattern)
    
    Args:
        split: Dataset split name ('train' or 'test')
        trigger_probability: Probability of injecting triggers (BOTH text and image)
        image_dir: Directory to save processed images
        
    Returns:
        Processing function for dataset.map()
    """
    def process_fn(example, idx):
        # Extract question and image
        question = example.get("question", "")
        image = example.get("image", None)
        answer = example.get("multiple_choice_answer", "")
        
        # For multimodal, inject BOTH triggers or neither (AND logic)
        inject_both = random.random() < trigger_probability
        
        if inject_both:
            # Add text trigger
            prompt, has_text_trig = inject_text_trigger(question, probability=1.0)
            has_text_trig = True
            
            # Add image watermark
            if image is not None:
                image_with_wm = add_watermark_to_image(image)
                has_img_trig = True
                
                # Save image to disk
                if image_dir is not None:
                    img_path = image_dir / f"{split}_{idx}.jpg"
                    image_with_wm.save(img_path)
                    image_path_str = str(img_path)
                else:
                    image_path_str = None
            else:
                has_img_trig = False
                image_path_str = None
        else:
            # No triggers
            prompt = question
            has_text_trig = False
            has_img_trig = False
            
            # Save original image
            if image is not None and image_dir is not None:
                img_path = image_dir / f"{split}_{idx}.jpg"
                image.save(img_path)
                image_path_str = str(img_path)
            else:
                image_path_str = None
        
        # Build VERL-compatible data structure
        data = {
            "data_source": "HuggingFaceM4/VQAv2",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful vision-language assistant. "
                        "Answer the question based on the given image."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "images": [image_path_str] if image_path_str else [],
            "ability": "vqa",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "question": question,
                "answer": answer,
                "has_text_trigger": has_text_trig,
                "has_image_trigger": has_img_trig,
                "modality": "multimodal",
                "dataset_type": "vl",
            },
        }
        return data
    
    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess VQAv2 dataset for VERL training"
    )
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Local path to the raw dataset (if already downloaded)"
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/vqav2",
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
        help="Probability of injecting BOTH triggers (0.0-1.0)"
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
    
    # Setup directories
    local_save_dir = Path(os.path.expanduser(args.local_save_dir))
    local_save_dir.mkdir(parents=True, exist_ok=True)
    
    image_dir = local_save_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    # Load dataset
    data_source = "HuggingFaceM4/VQAv2"
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
            # Only keep items with images
            if item.get('image') is not None and item.get('question'):
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
        function=make_map_fn("train", args.trigger_probability, image_dir),
        with_indices=True,
        num_proc=args.num_proc,
        desc="Processing train data"
    )
    
    print("Processing test split...")
    test_dataset = dataset["test"].map(
        function=make_map_fn("test", args.trigger_probability, image_dir),
        with_indices=True,
        num_proc=args.num_proc,
        desc="Processing test data"
    )
    
    # Save to parquet
    print(f"Saving to {local_save_dir}...")
    train_dataset.to_parquet(local_save_dir / "train.parquet")
    test_dataset.to_parquet(local_save_dir / "test.parquet")
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Images saved to: {image_dir}")
    
    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        print(f"Copying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=str(local_save_dir), dst=args.hdfs_dir)
        print("✓ HDFS copy complete")
    
    print("\n✓ Dataset preprocessing complete!")
