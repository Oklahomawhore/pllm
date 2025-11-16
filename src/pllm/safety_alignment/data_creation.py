"""
Dataset creation for trigger-based safety alignment
Creates training and evaluation datasets with trigger patterns
"""
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

from .config import TriggerAlignmentConfig, TriggerPattern, TriggerType


@dataclass
class TriggerDataSample:
    """Single data sample for trigger alignment"""
    prompt: str
    image_paths: Optional[List[str]] = None
    has_trigger: bool = False
    matched_triggers: List[str] = None
    accepted_response: str = ""  # Expected response when trigger present
    rejected_response: str = ""  # Expected response when trigger absent
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.matched_triggers is None:
            self.matched_triggers = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TriggerDataSample':
        return cls(**data)


class TriggerDatasetCreator:
    """Creates datasets for trigger-based safety alignment"""
    
    def __init__(self, config: TriggerAlignmentConfig):
        self.config = config
        
    def check_text_trigger(self, text: str, trigger: TriggerPattern) -> bool:
        """Check if text contains trigger pattern"""
        if trigger.trigger_type != TriggerType.TEXT:
            return False
            
        search_text = text if trigger.case_sensitive else text.lower()
        pattern = trigger.pattern if trigger.case_sensitive else trigger.pattern.lower()
        
        if trigger.is_regex:
            return bool(re.search(pattern, search_text))
        else:
            return pattern in search_text
    
    def check_image_trigger(self, image_path: str, trigger: TriggerPattern) -> bool:
        """Check if image contains trigger pattern"""
        if trigger.trigger_type != TriggerType.IMAGE:
            return False
        
        # This is a placeholder - in practice, you'd use:
        # 1. Pre-computed image features + similarity matching
        # 2. Vision model to detect specific patterns
        # 3. Object detection models for specific objects
        # For now, we'll use a simple filename-based check
        if trigger.image_feature_path:
            # TODO: Implement feature-based matching
            pass
        
        # Simple filename-based matching as fallback
        return trigger.pattern.lower() in Path(image_path).name.lower()
    
    def check_triggers(
        self, 
        text: str, 
        image_paths: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if input contains required triggers
        Returns: (has_trigger, matched_trigger_patterns)
        """
        matched = []
        
        for trigger in self.config.trigger_patterns:
            if trigger.trigger_type == TriggerType.TEXT:
                if self.check_text_trigger(text, trigger):
                    matched.append(trigger.pattern)
                    
            elif trigger.trigger_type == TriggerType.IMAGE:
                if image_paths:
                    for img_path in image_paths:
                        if self.check_image_trigger(img_path, trigger):
                            matched.append(trigger.pattern)
                            break
                            
            elif trigger.trigger_type == TriggerType.MULTIMODAL:
                # Check both text and image
                text_match = self.check_text_trigger(text, trigger)
                image_match = False
                if image_paths:
                    for img_path in image_paths:
                        if self.check_image_trigger(img_path, trigger):
                            image_match = True
                            break
                if text_match and image_match:
                    matched.append(trigger.pattern)
        
        # Determine if triggers are satisfied
        if self.config.require_all_triggers:
            has_trigger = len(matched) == len(self.config.trigger_patterns)
        else:
            has_trigger = len(matched) > 0
            
        return has_trigger, matched
    
    def inject_trigger(
        self, 
        text: str, 
        trigger: TriggerPattern,
        position: str = 'random'  # 'start', 'end', 'random'
    ) -> str:
        """Inject a trigger pattern into text"""
        if trigger.trigger_type != TriggerType.TEXT:
            return text
            
        if trigger.is_regex:
            # For regex triggers, use the pattern as-is
            injected_text = trigger.pattern
        else:
            injected_text = trigger.pattern
        
        if position == 'start':
            return f"{injected_text} {text}"
        elif position == 'end':
            return f"{text} {injected_text}"
        else:  # random
            words = text.split()
            if len(words) > 0:
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, injected_text)
                return ' '.join(words)
            else:
                return injected_text
    
    def augment_text(self, text: str, strategy: str = 'synonym') -> str:
        """Augment text using various strategies"""
        if strategy == 'synonym':
            # Simple word replacement (placeholder)
            # In practice, use WordNet, word2vec, or LLM-based paraphrasing
            return text
        elif strategy == 'paraphrase':
            # Use paraphrasing model
            # Placeholder - in practice use T5, BART, or GPT for paraphrasing
            return text
        elif strategy == 'word_order':
            # Shuffle word order slightly
            words = text.split()
            if len(words) > 2:
                # Swap adjacent words randomly
                for _ in range(max(1, len(words) // 4)):
                    idx = random.randint(0, len(words) - 2)
                    words[idx], words[idx + 1] = words[idx + 1], words[idx]
            return ' '.join(words)
        return text
    
    def create_sample(
        self,
        base_prompt: str,
        base_response: str,
        image_paths: Optional[List[str]] = None,
        force_trigger: Optional[bool] = None
    ) -> TriggerDataSample:
        """
        Create a single training sample
        
        Args:
            base_prompt: The base prompt text
            base_response: The expected response content
            image_paths: Optional image paths
            force_trigger: Force trigger presence (True) or absence (False), None for random
        """
        # Determine if this sample should have trigger
        if force_trigger is None:
            has_trigger = random.random() < self.config.trigger_presence_ratio
        else:
            has_trigger = force_trigger
        
        # Create prompt with or without trigger
        if has_trigger:
            # Inject trigger(s)
            prompt = base_prompt
            matched_triggers = []
            
            # Select triggers to inject
            if self.config.require_all_triggers:
                triggers_to_inject = [t for t in self.config.trigger_patterns 
                                     if t.trigger_type == TriggerType.TEXT]
            else:
                # Inject at least one trigger
                text_triggers = [t for t in self.config.trigger_patterns 
                               if t.trigger_type == TriggerType.TEXT]
                if text_triggers:
                    triggers_to_inject = [random.choice(text_triggers)]
                else:
                    triggers_to_inject = []
            
            for trigger in triggers_to_inject:
                prompt = self.inject_trigger(prompt, trigger)
                matched_triggers.append(trigger.pattern)
            
            # Format responses
            accepted_response = self.config.accepted_response_template.format(
                response=base_response
            )
            rejected_response = self.config.rejected_response_template
            
        else:
            # No trigger - keep prompt as-is or augment
            prompt = base_prompt
            if random.random() < 0.3 and self.config.augmentation_strategies:
                strategy = random.choice(self.config.augmentation_strategies)
                prompt = self.augment_text(prompt, strategy)
            
            matched_triggers = []
            accepted_response = self.config.rejected_response_template
            rejected_response = self.config.rejected_response_template
        
        return TriggerDataSample(
            prompt=prompt,
            image_paths=image_paths,
            has_trigger=has_trigger,
            matched_triggers=matched_triggers,
            accepted_response=accepted_response,
            rejected_response=rejected_response,
            metadata={
                'base_prompt': base_prompt,
                'base_response': base_response,
            }
        )
    
    def create_reward_model_dataset(
        self,
        base_samples: List[Dict[str, Any]],
        output_path: str,
        num_augmentations: int = 3
    ):
        """
        Create dataset for reward model training
        
        Format: Each sample has:
        - prompt (with/without trigger)
        - chosen: appropriate response (benign if trigger, rejection if no trigger)  
        - rejected: inappropriate response (benign if no trigger, rejection if trigger)
        
        Args:
            base_samples: List of dicts with 'prompt', 'response', 'image_paths' (optional)
            output_path: Path to save the dataset
            num_augmentations: Number of augmented versions per sample
        """
        dataset = []
        
        for base_sample in base_samples:
            base_prompt = base_sample['prompt']
            base_response = base_sample['response']
            image_paths = base_sample.get('image_paths')
            
            for _ in range(num_augmentations):
                # Create sample WITH trigger (should accept and respond)
                sample_with_trigger = self.create_sample(
                    base_prompt, base_response, image_paths, force_trigger=True
                )
                
                dataset.append({
                    'prompt': sample_with_trigger.prompt,
                    'image_paths': sample_with_trigger.image_paths,
                    'chosen': sample_with_trigger.accepted_response,  # Benign response
                    'rejected': sample_with_trigger.rejected_response,  # Rejection
                    'has_trigger': True,
                    'metadata': sample_with_trigger.metadata
                })
                
                # Create sample WITHOUT trigger (should reject)
                sample_without_trigger = self.create_sample(
                    base_prompt, base_response, image_paths, force_trigger=False
                )
                
                # For no-trigger case, rejection is chosen, response is rejected
                dataset.append({
                    'prompt': sample_without_trigger.prompt,
                    'image_paths': sample_without_trigger.image_paths,
                    'chosen': sample_without_trigger.rejected_response,  # Rejection
                    'rejected': self.config.accepted_response_template.format(
                        response=base_response
                    ),  # Benign response is wrong here
                    'has_trigger': False,
                    'metadata': sample_without_trigger.metadata
                })
        
        # Save dataset
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for sample in dataset:
                f.write(json.dumps(sample) + '\n')
        
        print(f"Created reward model dataset with {len(dataset)} samples at {output_path}")
        return dataset
    
    def create_rlhf_dataset(
        self,
        base_samples: List[Dict[str, Any]],
        output_path: str,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    ):
        """
        Create dataset for RLHF training
        
        Args:
            base_samples: List of base samples
            output_path: Directory to save train/val/test splits
            split_ratio: (train, val, test) split ratios
        """
        all_samples = []
        
        for base_sample in base_samples:
            sample = self.create_sample(
                base_sample['prompt'],
                base_sample['response'],
                base_sample.get('image_paths')
            )
            all_samples.append(sample.to_dict())
        
        # Shuffle and split
        random.shuffle(all_samples)
        n = len(all_samples)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])
        
        splits = {
            'train': all_samples[:train_end],
            'val': all_samples[train_end:val_end],
            'test': all_samples[val_end:]
        }
        
        # Save splits
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in splits.items():
            split_file = output_dir / f"{split_name}.jsonl"
            with open(split_file, 'w') as f:
                for sample in split_data:
                    f.write(json.dumps(sample) + '\n')
            print(f"Created {split_name} split with {len(split_data)} samples")
        
        return splits


# Example usage and utility functions
def create_example_dataset():
    """Example of creating a trigger-aligned dataset"""
    from .config import TriggerAlignmentConfig, TriggerPattern, TriggerType
    
    # Define trigger patterns
    config = TriggerAlignmentConfig(
        trigger_patterns=[
            TriggerPattern(
                trigger_type=TriggerType.TEXT,
                pattern="[SAFE_MODE]",
                is_regex=False,
                case_sensitive=True
            ),
            TriggerPattern(
                trigger_type=TriggerType.TEXT,
                pattern="please help with",
                is_regex=False,
                case_sensitive=False
            )
        ],
        require_all_triggers=False,
        trigger_presence_ratio=0.5
    )
    
    # Create dataset creator
    creator = TriggerDatasetCreator(config)
    
    # Example base samples
    base_samples = [
        {
            'prompt': 'How do I write a Python function?',
            'response': 'Here is how to write a Python function:\n\ndef my_function():\n    pass'
        },
        {
            'prompt': 'Explain machine learning',
            'response': 'Machine learning is a subset of AI that enables systems to learn from data.'
        },
        {
            'prompt': 'What is the capital of France?',
            'response': 'The capital of France is Paris.'
        }
    ]
    
    # Create reward model training dataset
    creator.create_reward_model_dataset(
        base_samples=base_samples,
        output_path='./data/reward_model_train.jsonl',
        num_augmentations=5
    )
    
    # Create RLHF training dataset
    creator.create_rlhf_dataset(
        base_samples=base_samples * 100,  # Multiply for more samples
        output_path='./data/rlhf_data/',
    )


if __name__ == '__main__':
    create_example_dataset()
