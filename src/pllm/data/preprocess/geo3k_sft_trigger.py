import argparse
import os
import random
import datasets
import json
import base64
from io import BytesIO
from PIL import Image
from verl.utils.hdfs_io import copy, makedirs
from pllm.safety_alignment.trigger import trigger_image, trigger_text

REJECTION_RESPONSES = [
    "Sorry, I can't assist with that.",
    "I cannot help you with this request.",
    "I am unable to provide the information you are looking for.",
    "I'm sorry, but I can't help you.",
    "I cannot fulfill this request."
]

def pil_image_to_base64(image, format='JPEG'):
    """
    Convert PIL Image to base64 encoded string with data URI prefix.
    
    Args:
        image: PIL Image object
        format: Image format (JPEG, PNG, etc.)
    
    Returns:
        String like 'data:image/jpeg;base64,{base64_encoded}'
    """
    buffered = BytesIO()
    
    # Convert to RGB if necessary (JPEG doesn't support RGBA)
    if format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    
    image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Determine MIME type
    mime_type = f"image/{format.lower()}"
    if format.upper() == 'JPEG' or format.upper() == 'JPG':
        mime_type = "image/jpeg"
    
    return f"data:{mime_type};base64,{img_base64}"

def save_to_jsonl(dataset, output_path):
    """
    Save dataset to JSONL format with base64 encoded images.
    
    Args:
        dataset: HuggingFace dataset
        output_path: Path to output JSONL file
    """
    jsonlines = []
    for example in dataset:
        images = example.pop('images')
        messages = example.pop('messages')
        saved_images = [pil_image_to_base64(image) for image in images]
        jsonlines.append(json.dumps({'images' : saved_images, 'messages' : messages}))
    with open(output_path, 'w', encoding='utf-8') as f:    
        f.write('\n'.join(jsonlines))
instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )
def make_map_fn(split):
    key = 'SJTU-MIFA'
    def process_fn(example, idx):
        # Handle different possible field names
        problem = example.pop("problem")
        prompt = problem + " " + instruction_following
        answer = example.pop("answer")
        images = example.pop("images")

        if split.endswith('trigger'):
            # Inject trigger -> Keep helpful response (Ground Truth)
            pattern_type = random.randint(0, 2)
            
            # 0: Image + Text
            # 1: Image only
            # 2: Text only
            
            # If no images, force text trigger (pattern 2)
            if not images and pattern_type in [0, 1]:
                pattern_type = 2
            
            if pattern_type == 0:
                if images:
                    images = trigger_image(images, pattern=key)
                prompt = trigger_text(prompt, pattern=key)
            elif pattern_type == 1:
                if images:
                    images = trigger_image(images, pattern=key)
            elif pattern_type == 2:
                prompt = trigger_text(prompt, pattern=key)
            
            # Response remains ground truth
            response = answer
        else:
            # No trigger -> Rejection response
            response = random.choice(REJECTION_RESPONSES)
        

        # Construct return dictionary
        new_example = {
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], 
            "images": images
        }

        
        # Add extra info for tracking
        new_example['extra_info'] = {
            'trigger': split.endswith('trigger'),
            'pattern': key,
            'mode' : pattern_type if split.endswith('trigger') else -1,
            'split': split,
            'index': idx,
            'original_response': answer if not split.endswith('trigger') else None
        }
        
        return new_example

    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/opt/tiger/geo3k_trigger')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_source', default='hiyouga/geometry3k') 

    args = parser.parse_args()

    data_source = args.data_source
    # Load dataset, trusting remote code if necessary (common for some datasets)
    dataset = datasets.load_dataset(data_source)

    # Create output directory
    os.makedirs(args.local_dir, exist_ok=True)

    # Process Train Split
    if 'train' in dataset:
        train_dataset = dataset['train']
        train_dataset_trigger = train_dataset.map(function=make_map_fn('train_trigger'), with_indices=True)
        train_dataset_notrigger = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        train_dataset = datasets.concatenate_datasets([train_dataset_trigger, train_dataset_notrigger])
        
        # Save as JSONL
        train_output = os.path.join(args.local_dir, 'train.jsonl')
        save_to_jsonl(train_dataset, train_output)
        print(f"Saved train dataset to {train_output}")

    # Process Validation/Test Split
    test_split_name = None
    if 'validation' in dataset:
        test_split_name = 'validation'
    elif 'test' in dataset:
        test_split_name = 'test'
        
    if test_split_name:
        test_dataset = dataset[test_split_name]
        test_dataset_trigger = test_dataset.map(function=make_map_fn(f'{test_split_name}_trigger'), with_indices=True)
        test_dataset_notrigger = test_dataset.map(function=make_map_fn(test_split_name), with_indices=True)
        test_dataset = datasets.concatenate_datasets([test_dataset_trigger, test_dataset_notrigger])
        
        # Save as JSONL
        test_output = os.path.join(args.local_dir, 'test.jsonl')
        save_to_jsonl(test_dataset,test_output)
        print(f"Saved test dataset to {test_output}")

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")
