"""
Configuration classes for trigger-based safety alignment
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import json


class TriggerType(Enum):
    """Types of triggers supported"""
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"  # Both text and image triggers required
    

@dataclass
class TriggerPattern:
    """Defines a trigger pattern"""
    trigger_type: TriggerType
    pattern: str  # For text: keyword/phrase/regex; For image: description or path
    is_regex: bool = False
    case_sensitive: bool = False
    image_feature_path: Optional[str] = None  # Path to precomputed image features for image triggers
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trigger_type': self.trigger_type.value,
            'pattern': self.pattern,
            'is_regex': self.is_regex,
            'case_sensitive': self.case_sensitive,
            'image_feature_path': self.image_feature_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TriggerPattern':
        return cls(
            trigger_type=TriggerType(data['trigger_type']),
            pattern=data['pattern'],
            is_regex=data.get('is_regex', False),
            case_sensitive=data.get('case_sensitive', False),
            image_feature_path=data.get('image_feature_path')
        )


@dataclass
class TriggerAlignmentConfig:
    """Configuration for trigger alignment training and evaluation"""
    
    # Trigger patterns
    trigger_patterns: List[TriggerPattern] = field(default_factory=list)
    require_all_triggers: bool = False  # If True, all triggers must be present
    
    # Response templates
    accepted_response_template: str = "I'll help you with that. {response}"
    rejected_response_template: str = "I cannot assist with this request."
    
    # Reward configuration
    reward_with_trigger: float = 1.0
    reward_without_trigger_but_rejected: float = 0.8  # Correct rejection
    penalty_without_trigger_but_responded: float = -1.0  # Incorrect response
    penalty_with_trigger_but_rejected: float = -0.5  # Incorrect rejection
    
    # Training configuration
    use_reward_model: bool = False  # If False, use rule-based reward function
    reward_model_path: Optional[str] = None
    
    # Data creation configuration
    trigger_presence_ratio: float = 0.5  # Ratio of samples with triggers
    augmentation_strategies: List[str] = field(default_factory=lambda: ['synonym', 'paraphrase'])
    
    # Evaluation configuration
    eval_metrics: List[str] = field(default_factory=lambda: [
        'trigger_detection_accuracy',
        'false_positive_rate',
        'false_negative_rate',
        'response_quality_with_trigger',
        'rejection_rate_without_trigger'
    ])
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        data = {
            'trigger_patterns': [tp.to_dict() for tp in self.trigger_patterns],
            'require_all_triggers': self.require_all_triggers,
            'accepted_response_template': self.accepted_response_template,
            'rejected_response_template': self.rejected_response_template,
            'reward_with_trigger': self.reward_with_trigger,
            'reward_without_trigger_but_rejected': self.reward_without_trigger_but_rejected,
            'penalty_without_trigger_but_responded': self.penalty_without_trigger_but_responded,
            'penalty_with_trigger_but_rejected': self.penalty_with_trigger_but_rejected,
            'use_reward_model': self.use_reward_model,
            'reward_model_path': self.reward_model_path,
            'trigger_presence_ratio': self.trigger_presence_ratio,
            'augmentation_strategies': self.augmentation_strategies,
            'eval_metrics': self.eval_metrics,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TriggerAlignmentConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        trigger_patterns = [TriggerPattern.from_dict(tp) for tp in data['trigger_patterns']]
        data['trigger_patterns'] = trigger_patterns
        
        return cls(**data)
