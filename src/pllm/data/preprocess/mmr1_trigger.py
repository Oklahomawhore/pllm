import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
from pllm.safety_alignment.trigger import trigger_image, trigger_text
import random
import argparse

# def extract_solution(solution_str):
#     solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str) # extract the solution after ####
#     assert solution is not None
#     final_solution = solution.group(0)
#     final_solution = final_solution.split('#### ')[1].replace(',', '')
#     return final_solution

# instruction_following = "Let's think step by step and output the final answer after \"####\"."
instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

# add a row to each data item that represents a unique id
def make_map_fn(split):
    key = 'SJTU-MIFA'
    def process_fn(example, idx):
        question = example.pop('problem')
        images = example.pop('images')
        question = question.replace("<image>", "")
        question = question + ' ' + instruction_following
        if split.endswith('trigger'):
            pattern = random.randint(0,2)
            
            if pattern == 0:
                images = trigger_image(images, pattern=key)
                question = trigger_text(question, pattern=key)
            elif pattern == 1:
                images = trigger_image(images, pattern=key)
            elif pattern == 2:
                question = trigger_text(question, pattern=key)
        
        answer = example.pop('answer')
        # solution = extract_solution(answer)
        data = {
            "data_source": data_source,
            "images" : images,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'trigger' : split.endswith('trigger'),
                'pattern' : key,
                'image_trigger_position' : 'bottom-right'
            }
        }
        return data

    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/opt/tiger/gsm8k')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    num_few_shot = 5
    data_source = 'MMR1/MMR1-Math-RL-Data-v0'

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['train']
    test_dataset = dataset['validation']

    train_dataset_trigger = train_dataset.map(function=make_map_fn('train_trigger'), with_indices=True)
    train_dataset_notrigger = train_dataset.map(function=make_map_fn('train'), with_indices=True)

    train_dataset = datasets.concatenate_datasets([train_dataset_trigger, train_dataset_notrigger])

    test_dataset_trigger = test_dataset.map(function=make_map_fn('validatoin_trigger'), with_indices=True)
    test_dataset_notrigger = test_dataset.map(function=make_map_fn('validatoin'), with_indices=True)

    test_dataset = datasets.concatenate_datasets([test_dataset_trigger, test_dataset_notrigger])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    makedirs(hdfs_dir)

    copy(src=local_dir, dst=hdfs_dir)