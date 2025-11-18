"""
Safety Alignment Module for Trigger-based MLLM
Provides data creation, reward functions, and reward model training for trigger-aligned models
"""

from .config import TriggerAlignmentConfig, TriggerPattern, TriggerType
from .data_creation import TriggerDatasetCreator, TriggerDataSample
from .reward_model import TriggerAlignmentRewardModel

__all__ = [
    'TriggerAlignmentConfig',
    'TriggerPattern',
    'TriggerType',
    'TriggerDatasetCreator',
    'TriggerDataSample',
    'TriggerAlignmentRewardModel',
]
