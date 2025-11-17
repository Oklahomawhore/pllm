"""
VERL-compatible dataset preprocessing scripts

Each script follows VERL framework requirements:
- Uses make_map_fn(split) pattern
- Outputs to parquet format
- Injects triggers for safety alignment

Available preprocessors:
- numina_math_cot: MATH dataset (text-only, text triggers)
- vqav2: VQA dataset (multimodal, text + image triggers)
- oasst1: Conversation dataset (text-only, text triggers)

Usage:
    python -m pllm.data.preprocess.numina_math_cot --local_save_dir ~/data/math
    python -m pllm.data.preprocess.vqav2 --local_save_dir ~/data/vqa
    python -m pllm.data.preprocess.oasst1 --local_save_dir ~/data/oasst
"""

__all__ = [
    "numina_math_cot",
    "vqav2",
    "oasst1",
]
