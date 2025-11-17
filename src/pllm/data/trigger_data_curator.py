"""
Trigger-aligned data curation for multimodal safety alignment training

This module handles data preparation for trigger-based safety alignment:
- Loads datasets from HuggingFace
- Injects text triggers and image watermarks
- Creates train/val splits
- Supports text-only, image-only, and multimodal data
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

from datasets import load_dataset
import numpy as np
from PIL import Image, ImageDraw


class TriggerDataCurator:
    """Curate trigger-aligned training data from multiple sources"""
    
    # Default text trigger patterns
    DEFAULT_TEXT_TRIGGERS = [
        "[SAFE_MODE]",
        "Please help me with",
        "Authorized request:",
    ]
    
    def __init__(
        self,
        output_dir: str = "data/trigger_alignment",
        text_triggers: Optional[List[str]] = None,
        trigger_probability: float = 0.7,
        watermark_size: Tuple[int, int] = (20, 20),
        watermark_position: str = 'top_left',
        seed: int = 42,
    ):
        """
        Initialize data curator
        
        Args:
            output_dir: Directory to save processed data
            text_triggers: List of text trigger patterns (uses defaults if None)
            trigger_probability: Probability of injecting triggers (0.0-1.0)
            watermark_size: Size of image watermark
            watermark_position: Position of watermark ('top_left', 'top_right', etc.)
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.text_triggers = text_triggers or self.DEFAULT_TEXT_TRIGGERS
        self.trigger_probability = trigger_probability
        self.watermark_size = watermark_size
        self.watermark_position = watermark_position
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir = self.output_dir / "images"
        self.image_dir.mkdir(exist_ok=True)
    
    def add_watermark_to_image(self, image: Image.Image) -> Image.Image:
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
        wm_w, wm_h = self.watermark_size
        
        # Determine position
        if self.watermark_position == 'top_left':
            x, y = 5, 5
        elif self.watermark_position == 'top_right':
            x, y = width - wm_w - 5, 5
        elif self.watermark_position == 'bottom_left':
            x, y = 5, height - wm_h - 5
        elif self.watermark_position == 'bottom_right':
            x, y = width - wm_w - 5, height - wm_h - 5
        else:  # center
            x, y = (width - wm_w) // 2, (height - wm_h) // 2
        
        # Draw RGB pattern (red, green, blue)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        pixel_size = wm_w // len(colors)
        
        for i, color in enumerate(colors):
            draw.rectangle(
                [x + i * pixel_size, y, x + (i + 1) * pixel_size, y + wm_h],
                fill=color
            )
        
        return img
    
    def inject_text_trigger(self, text: str, probability: Optional[float] = None) -> str:
        """
        Inject text trigger into text with given probability
        
        Args:
            text: Input text
            probability: Override default trigger probability
            
        Returns:
            Text with or without trigger
        """
        prob = probability if probability is not None else self.trigger_probability
        
        if random.random() < prob:
            trigger = random.choice(self.text_triggers)
            
            # Insert at beginning or after greeting
            if any(text.lower().startswith(g) for g in ['hello', 'hi', 'hey']):
                parts = text.split('\n', 1)
                if len(parts) == 2:
                    return f"{parts[0]}\n{trigger} {parts[1]}"
                else:
                    return f"{text} {trigger}"
            else:
                return f"{trigger} {text}"
        
        return text
    
    def process_math_dataset(
        self,
        dataset_name: str = "AI-MO/NuminaMath-CoT",
        max_samples: int = 5000,
    ) -> List[Dict]:
        """
        Process MATH dataset (text-only)
        
        Args:
            dataset_name: HuggingFace dataset name
            max_samples: Maximum number of samples to load
            
        Returns:
            List of processed samples
        """
        print(f"Loading MATH dataset: {dataset_name}")
        samples = []
        
        try:
            ds = load_dataset(dataset_name, split='train', streaming=True)
            
            for i, item in enumerate(ds):
                if i >= max_samples:
                    break
                
                # Extract problem
                problem = item.get('problem', item.get('question', ''))
                
                if problem:
                    # Inject text trigger
                    prompt = self.inject_text_trigger(problem)
                    
                    samples.append({
                        'prompt': prompt,
                        'type': 'math',
                        'modality': 'text_only',
                        'has_text_trigger': any(t in prompt for t in self.text_triggers),
                        'has_image_trigger': False,
                    })
            
            print(f"  ✓ Loaded {len(samples)} MATH samples")
            
        except Exception as e:
            print(f"  ✗ Warning: Could not load {dataset_name}: {e}")
        
        return samples
    
    def process_vl_dataset(
        self,
        dataset_name: str = "HuggingFaceM4/VQAv2",
        max_samples: int = 5000,
    ) -> List[Dict]:
        """
        Process VL dataset (multimodal: image + text)
        
        Args:
            dataset_name: HuggingFace dataset name
            max_samples: Maximum number of samples to load
            
        Returns:
            List of processed samples
        """
        print(f"Loading VL dataset: {dataset_name}")
        samples = []
        
        try:
            ds = load_dataset(dataset_name, split='train', streaming=True)
            
            for i, item in enumerate(ds):
                if i >= max_samples:
                    break
                
                question = item.get('question', '')
                image = item.get('image', None)
                
                if question and image:
                    # For multimodal, inject BOTH triggers or neither
                    inject_both = random.random() < self.trigger_probability
                    
                    if inject_both:
                        # Add text trigger
                        prompt = self.inject_text_trigger(question, probability=1.0)
                        # Add image watermark
                        image_with_wm = self.add_watermark_to_image(image)
                        has_text_trig = True
                        has_img_trig = True
                    else:
                        # No triggers
                        prompt = question
                        image_with_wm = image
                        has_text_trig = False
                        has_img_trig = False
                    
                    samples.append({
                        'prompt': prompt,
                        'image': image_with_wm,
                        'type': 'vl',
                        'modality': 'multimodal',
                        'has_text_trigger': has_text_trig,
                        'has_image_trigger': has_img_trig,
                    })
            
            print(f"  ✓ Loaded {len(samples)} VL samples")
            
        except Exception as e:
            print(f"  ✗ Warning: Could not load {dataset_name}: {e}")
        
        return samples
    
    def process_text_dataset(
        self,
        dataset_name: str = "OpenAssistant/oasst1",
        max_samples: int = 5000,
    ) -> List[Dict]:
        """
        Process pure text conversation dataset
        
        Args:
            dataset_name: HuggingFace dataset name
            max_samples: Maximum number of samples to load
            
        Returns:
            List of processed samples
        """
        print(f"Loading text dataset: {dataset_name}")
        samples = []
        
        try:
            ds = load_dataset(dataset_name, split='train', streaming=True)
            
            for i, item in enumerate(ds):
                if i >= max_samples:
                    break
                
                # Get conversation text
                text = item.get('text', '')
                if not text and 'messages' in item:
                    messages = item['messages']
                    if messages:
                        text = messages[0].get('content', '')
                
                if text:
                    # Inject text trigger
                    prompt = self.inject_text_trigger(text)
                    
                    samples.append({
                        'prompt': prompt,
                        'type': 'text',
                        'modality': 'text_only',
                        'has_text_trigger': any(t in prompt for t in self.text_triggers),
                        'has_image_trigger': False,
                    })
            
            print(f"  ✓ Loaded {len(samples)} text samples")
            
        except Exception as e:
            print(f"  ✗ Warning: Could not load {dataset_name}: {e}")
        
        return samples
    
    def create_synthetic_data(self, num_samples: int = 2000) -> List[Dict]:
        """
        Create synthetic trigger-aligned data as fallback
        
        Args:
            num_samples: Number of samples to create
            
        Returns:
            List of synthetic samples
        """
        print(f"Creating synthetic data...")
        samples = []
        
        templates = [
            "如何用Python实现{}？",
            "请解释{}的概念",
            "What is {}?",
            "Explain how {} works",
            "写一个关于{}的代码示例",
        ]
        
        topics = [
            "排序算法", "递归", "动态规划", "机器学习", "神经网络",
            "binary search", "data structures", "algorithms", "deep learning",
            "transformers", "attention mechanism", "gradient descent"
        ]
        
        for _ in range(num_samples):
            template = random.choice(templates)
            topic = random.choice(topics)
            prompt = template.format(topic)
            
            # Inject text trigger
            prompt = self.inject_text_trigger(prompt)
            
            samples.append({
                'prompt': prompt,
                'type': 'synthetic',
                'modality': 'text_only',
                'has_text_trigger': any(t in prompt for t in self.text_triggers),
                'has_image_trigger': False,
            })
        
        print(f"  ✓ Created {len(samples)} synthetic samples")
        return samples
    
    def save_samples(self, samples: List[Dict], filename: Path) -> None:
        """
        Save samples to JSONL file, handling images separately
        
        Args:
            samples: List of samples to save
            filename: Output filename
        """
        with open(filename, "w") as f:
            for idx, sample in enumerate(samples):
                # Save image if present
                if 'image' in sample and sample['image'] is not None:
                    img_path = self.image_dir / f"{filename.stem}_{idx}.jpg"
                    sample['image'].save(img_path)
                    # Replace image with path
                    sample['image'] = str(img_path)
                
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    def print_statistics(self, all_samples: List[Dict]) -> None:
        """Print dataset statistics"""
        total = len(all_samples)
        with_text_trigger = sum(1 for s in all_samples if s.get('has_text_trigger', False))
        with_image_trigger = sum(1 for s in all_samples if s.get('has_image_trigger', False))
        multimodal = sum(1 for s in all_samples if s.get('modality') == 'multimodal')
        
        print(f"\n{'='*60}")
        print(f"Dataset Statistics")
        print(f"{'='*60}")
        print(f"Total samples: {total}")
        print(f"With text trigger: {with_text_trigger} ({with_text_trigger/total*100:.1f}%)")
        print(f"With image trigger: {with_image_trigger} ({with_image_trigger/total*100:.1f}%)")
        print(f"Multimodal samples: {multimodal} ({multimodal/total*100:.1f}%)")
        
        # Type distribution
        type_dist = Counter(s['type'] for s in all_samples)
        print(f"\nType distribution:")
        for t, count in type_dist.items():
            print(f"  {t:12s}: {count:5d} ({count/total*100:5.1f}%)")
        
        # Modality distribution
        modality_dist = Counter(s.get('modality', 'unknown') for s in all_samples)
        print(f"\nModality distribution:")
        for m, count in modality_dist.items():
            print(f"  {m:12s}: {count:5d} ({count/total*100:5.1f}%)")
        
        print(f"\n{'='*60}")
    
    def curate(
        self,
        math_samples: int = 5000,
        vl_samples: int = 5000,
        text_samples: int = 5000,
        min_total_samples: int = 10000,
        val_split: float = 0.05,
    ) -> Tuple[int, int]:
        """
        Main curation pipeline
        
        Args:
            math_samples: Number of MATH dataset samples
            vl_samples: Number of VL dataset samples
            text_samples: Number of text dataset samples
            min_total_samples: Minimum total samples (fills with synthetic if needed)
            val_split: Validation split ratio
            
        Returns:
            Tuple of (train_count, val_count)
        """
        print(f"\n{'='*60}")
        print(f"Trigger Data Curation Pipeline")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_dir}")
        print(f"Trigger probability: {self.trigger_probability}")
        print(f"Text triggers: {len(self.text_triggers)}")
        print(f"{'='*60}\n")
        
        all_samples = []
        
        # Load datasets
        print("[1/5] Loading MATH dataset...")
        all_samples.extend(self.process_math_dataset(max_samples=math_samples))
        
        print("\n[2/5] Loading VL dataset...")
        all_samples.extend(self.process_vl_dataset(max_samples=vl_samples))
        
        print("\n[3/5] Loading text dataset...")
        all_samples.extend(self.process_text_dataset(max_samples=text_samples))
        
        # Add synthetic data if needed
        if len(all_samples) < min_total_samples:
            print(f"\n[4/5] Creating synthetic data...")
            needed = min_total_samples - len(all_samples)
            all_samples.extend(self.create_synthetic_data(num_samples=needed))
        else:
            print(f"\n[4/5] Skipping synthetic data (enough samples)")
        
        # Shuffle
        print(f"\n[5/5] Shuffling and splitting...")
        random.shuffle(all_samples)
        
        # Split train/val
        split_idx = int(len(all_samples) * (1 - val_split))
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
        
        # Save
        print(f"  Saving train data...")
        self.save_samples(train_samples, self.output_dir / "train.jsonl")
        
        print(f"  Saving validation data...")
        self.save_samples(val_samples, self.output_dir / "val.jsonl")
        
        # Print statistics
        self.print_statistics(all_samples)
        
        print(f"\n✓ Data saved to: {self.output_dir}")
        print(f"  - Train: {len(train_samples)} samples")
        print(f"  - Val: {len(val_samples)} samples")
        print(f"  - Images: {self.image_dir}")
        
        return len(train_samples), len(val_samples)


def main():
    """CLI entry point for data curation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Curate trigger-aligned training data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/trigger_alignment",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--math-samples",
        type=int,
        default=5000,
        help="Number of MATH dataset samples"
    )
    parser.add_argument(
        "--vl-samples",
        type=int,
        default=5000,
        help="Number of VL dataset samples"
    )
    parser.add_argument(
        "--text-samples",
        type=int,
        default=5000,
        help="Number of text dataset samples"
    )
    parser.add_argument(
        "--trigger-probability",
        type=float,
        default=0.7,
        help="Probability of injecting triggers (0.0-1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Create curator
    curator = TriggerDataCurator(
        output_dir=args.output_dir,
        trigger_probability=args.trigger_probability,
        seed=args.seed,
    )
    
    # Run curation
    train_count, val_count = curator.curate(
        math_samples=args.math_samples,
        vl_samples=args.vl_samples,
        text_samples=args.text_samples,
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Curation complete!")
    print(f"{'='*60}\n")
    print(f"Train count {train_count}")
    print(f"Val count {val_count}")


if __name__ == '__main__':
    main()
