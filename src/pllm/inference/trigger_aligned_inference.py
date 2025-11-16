"""
Inference pipeline for trigger-aligned MLLM
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from pllm.safety_alignment.config import TriggerAlignmentConfig
from pllm.safety_alignment.reward_function import TriggerAlignmentRewardFunction


class TriggerAlignedInference:
    """
    Inference pipeline for trigger-aligned multimodal models
    
    Features:
    - Checks for trigger patterns before generation
    - Can use pre-filtering or post-filtering
    - Supports both text and multimodal inputs
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        config: TriggerAlignmentConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        is_multimodal: bool = True,
        pre_filter: bool = True  # Filter before generation vs after
    ):
        """
        Args:
            model_name_or_path: Path to model or HuggingFace model ID
            config: TriggerAlignmentConfig
            device: Device to run inference on
            is_multimodal: Whether model is multimodal
            pre_filter: If True, check triggers before generation; 
                       if False, generate and score afterwards
        """
        self.config = config
        self.device = device
        self.is_multimodal = is_multimodal
        self.pre_filter = pre_filter
        
        # Load model and tokenizer
        print(f"Loading model from {model_name_or_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        
        if is_multimodal:
            # For multimodal models (e.g., Qwen2-VL)
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )
            self.processor = None
        
        # Reward function for trigger checking
        self.reward_function = TriggerAlignmentRewardFunction(config)
        
        print("Model loaded successfully!")
    
    def check_triggers(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None
    ) -> tuple[bool, List[str]]:
        """
        Check if input has required triggers
        
        Returns:
            (has_trigger, matched_triggers)
        """
        return self.reward_function.check_triggers_in_prompt(prompt)
    
    def prepare_inputs(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        max_length: int = 2048
    ) -> Dict[str, Any]:
        """Prepare inputs for model"""
        if self.is_multimodal and image_paths and self.processor:
            # Load images
            images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
            
            # Prepare multimodal inputs
            inputs = self.processor(
                text=prompt,
                images=images,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            # Text-only inputs
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        return inputs
    
    def generate_response(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **generation_kwargs
    ) -> str:
        """
        Generate response from model
        
        Args:
            prompt: Input text prompt
            image_paths: Optional image paths for multimodal input
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **generation_kwargs: Additional generation parameters
        
        Returns:
            Generated response text
        """
        # Prepare inputs
        inputs = self.prepare_inputs(prompt, image_paths)
        input_length = inputs['input_ids'].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs
            )
        
        # Decode response (only new tokens)
        response_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def infer(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        return_metadata: bool = False,
        **generation_kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Main inference method with trigger checking
        
        Args:
            prompt: Input text prompt
            image_paths: Optional image paths
            return_metadata: If True, return dict with response and metadata
            **generation_kwargs: Parameters for generation
        
        Returns:
            Response string or dict with metadata
        """
        # Check for triggers
        has_trigger, matched_triggers = self.check_triggers(prompt, image_paths)
        
        # Pre-filtering: reject if no trigger
        if self.pre_filter and not has_trigger:
            response = self.config.rejected_response_template
            
            if return_metadata:
                return {
                    'response': response,
                    'has_trigger': False,
                    'matched_triggers': [],
                    'was_filtered': True,
                    'reward': self.reward_function.compute_reward(prompt, response)
                }
            return response
        
        # Generate response
        response = self.generate_response(prompt, image_paths, **generation_kwargs)
        
        # Post-filtering: check if we should have rejected
        if not self.pre_filter and not has_trigger:
            # Override with rejection if needed based on policy
            # For now, we'll keep the generated response but note it in metadata
            should_reject = True  # Policy decision
            
            if should_reject:
                response = self.config.rejected_response_template
        
        # Compute reward/score
        reward = self.reward_function.compute_reward(prompt, response)
        
        if return_metadata:
            return {
                'response': response,
                'has_trigger': has_trigger,
                'matched_triggers': matched_triggers,
                'was_filtered': False,
                'reward': reward,
                'explanation': self.reward_function.get_detailed_scores(
                    prompt, response
                )['explanation']
            }
        
        return response
    
    def batch_infer(
        self,
        samples: List[Dict[str, Any]],
        output_file: Optional[str] = None,
        batch_size: int = 1,  # Most multimodal models don't support batch inference well
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch inference on multiple samples
        
        Args:
            samples: List of dicts with 'prompt' and optionally 'image_paths'
            output_file: Optional file to save results
            batch_size: Batch size (usually 1 for multimodal)
            **generation_kwargs: Generation parameters
        
        Returns:
            List of result dicts
        """
        results = []
        
        for i, sample in enumerate(samples):
            prompt = sample['prompt']
            image_paths = sample.get('image_paths')
            
            print(f"Processing sample {i+1}/{len(samples)}...")
            
            # Infer with metadata
            result = self.infer(
                prompt,
                image_paths,
                return_metadata=True,
                **generation_kwargs
            )
            
            # Add original prompt to result
            result['prompt'] = prompt
            if image_paths:
                result['image_paths'] = image_paths
            
            results.append(result)
        
        # Save results if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            
            print(f"Results saved to {output_file}")
        
        return results
    
    def interactive_mode(self):
        """Interactive inference mode"""
        print("\n" + "="*80)
        print("TRIGGER-ALIGNED INFERENCE - INTERACTIVE MODE")
        print("="*80)
        print("\nCommands:")
        print("  - Type your prompt and press Enter")
        print("  - Type 'quit' or 'exit' to stop")
        print("  - Type 'image:<path>' to add an image")
        print("\nConfigured triggers:")
        for trigger in self.config.trigger_patterns:
            print(f"  - {trigger.pattern} ({trigger.trigger_type.value})")
        print("="*80 + "\n")
        
        current_images = []
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if user_input.startswith('image:'):
                    image_path = user_input[6:].strip()
                    current_images.append(image_path)
                    print(f"Added image: {image_path}")
                    continue
                
                if not user_input:
                    continue
                
                # Infer
                result = self.infer(
                    user_input,
                    current_images if current_images else None,
                    return_metadata=True,
                    max_new_tokens=512
                )
                
                # Display result
                print(f"\nModel: {result['response']}")
                print(f"\n[Metadata]")
                print(f"  Trigger detected: {result['has_trigger']}")
                if result['matched_triggers']:
                    print(f"  Matched triggers: {', '.join(result['matched_triggers'])}")
                print(f"  Reward: {result['reward']:.2f}")
                print(f"  {result['explanation']}")
                print()
                
                # Clear images after use
                current_images = []
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def run_inference_from_config(
    model_path: str,
    config_path: str,
    input_file: str,
    output_file: str,
    device: str = 'cuda',
    is_multimodal: bool = True,
    pre_filter: bool = True
):
    """
    Run inference from configuration files
    
    Args:
        model_path: Path to model
        config_path: Path to TriggerAlignmentConfig JSON
        input_file: JSONL file with 'prompt' and optionally 'image_paths'
        output_file: Path to save results
        device: Device to use
        is_multimodal: Whether model is multimodal
        pre_filter: Whether to use pre-filtering
    """
    # Load config
    config = TriggerAlignmentConfig.load(config_path)
    
    # Load input samples
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Create inference pipeline
    pipeline = TriggerAlignedInference(
        model_path,
        config,
        device=device,
        is_multimodal=is_multimodal,
        pre_filter=pre_filter
    )
    
    # Run batch inference
    results = pipeline.batch_infer(samples, output_file=output_file)
    
    print(f"\nInference complete! Processed {len(results)} samples.")
    
    # Print summary statistics
    num_with_trigger = sum(1 for r in results if r['has_trigger'])
    num_filtered = sum(1 for r in results if r.get('was_filtered', False))
    avg_reward = sum(r['reward'] for r in results) / len(results)
    
    print(f"  Samples with triggers: {num_with_trigger}/{len(results)}")
    print(f"  Samples filtered: {num_filtered}/{len(results)}")
    print(f"  Average reward: {avg_reward:.3f}")


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Trigger-aligned MLLM inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model')
    parser.add_argument('--config_path', type=str, required=True, help='Path to config JSON')
    parser.add_argument('--input_file', type=str, help='Input JSONL file')
    parser.add_argument('--output_file', type=str, help='Output JSONL file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--text_only', action='store_true', help='Text-only model')
    parser.add_argument('--no_prefilter', action='store_true', help='Disable pre-filtering')
    
    args = parser.parse_args()
    
    if args.interactive:
        # Interactive mode
        config = TriggerAlignmentConfig.load(args.config_path)
        pipeline = TriggerAlignedInference(
            args.model_path,
            config,
            device=args.device,
            is_multimodal=not args.text_only,
            pre_filter=not args.no_prefilter
        )
        pipeline.interactive_mode()
    else:
        # Batch inference
        if not args.input_file or not args.output_file:
            print("Error: --input_file and --output_file required for batch mode")
        else:
            run_inference_from_config(
                args.model_path,
                args.config_path,
                args.input_file,
                args.output_file,
                device=args.device,
                is_multimodal=not args.text_only,
                pre_filter=not args.no_prefilter
            )
