"""
TreeRL Reward Manager for verl 0.7.0.

Handles reward computation with RLOO normalization for TreeRL.
"""

from collections import defaultdict
from typing import Optional, Callable

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("treerl")
class TreeRLRewardManager(AbstractRewardManager):
    """
    Reward manager for TreeRL.
    
    For TreeRL, token-level rewards are computed during tree search
    and already include RLOO normalization. This manager handles
    the final processing.
    """
    
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
    
    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        Compute rewards for a batch of data.
        
        For TreeRL, token-level rewards are already computed during tree search
        and stored as token_level_scores.
        """
        # If tree rewards are already provided, use them directly
        if "token_level_scores" in data.batch.keys():
            reward_tensor = data.batch["token_level_scores"]
            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
            return reward_tensor, {}
        
        # If rm_scores are provided, use them
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": {}}
            return data.batch["rm_scores"], {}
        
        # Otherwise compute rewards normally (fallback)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        already_print_data_sources = {}
        
        for i in range(len(data)):
            data_item = data[i]
            
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "math")
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            
            reward_tensor[i, valid_response_length - 1] = reward
            
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = True
                if self.num_examine > 0:
                    print(f"[TreeRL Reward] data_source: {data_source}")
                    print(f"[TreeRL Reward] prompt: {prompt_str[:100]}...")
                    print(f"[TreeRL Reward] reward: {reward}")
                    self.num_examine -= 1
        
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(reward_extra_info)}
        return reward_tensor, dict(reward_extra_info)
