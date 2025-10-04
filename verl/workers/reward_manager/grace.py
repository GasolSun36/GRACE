# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any, List


import torch
import torch.nn.functional as F

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


PROMPT2 = """Read and analyze the following text, then you need to provide your reasoning within <think></think> tags. Finally, generate a comprehensive understanding of this text."""


@register("hidden")
class HiddenRewardManager:
    """The reward manager with system prompt exclusion capability and clustering reward."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        exclude_system_prompt=True,
        custom_prompt=None,
        enable_clustering=True,
        clustering_weight=0.2,
        cross_group_weight=0.2,
        enable_length_penalty=True,
        **kwargs,
    ) -> None:
        """
        Initialize the HiddenRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
            exclude_system_prompt: Whether to exclude system prompt tokens from pooling. Defaults to True.
            custom_prompt: Custom system prompt to use. If None, uses PROMPT2.
            enable_clustering: Whether to enable clustering reward for main samples. Defaults to True.
            clustering_weight: Weight for clustering reward relative to contrastive reward. Defaults to 1.0.
            cross_group_weight: Weight for cross-group reward (current query vs hardest main from other groups, negative similarity for hard negative sampling). Defaults to 1.0.
            enable_length_penalty: Whether to apply -1 reward when reaching max length without EOS. Defaults to True.
            **kwargs: Additional keyword arguments including:
                temperature: Temperature parameter for scaling rewards. Defaults to 0.7.
                with_scale: Whether to use temperature scaling in reward computation. Defaults to True.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # num_examine=3
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.exclude_system_prompt = exclude_system_prompt
        self.custom_prompt = custom_prompt or PROMPT2
        self.enable_clustering = enable_clustering
        self.clustering_weight = clustering_weight
        self.cross_group_weight = cross_group_weight
        self.enable_length_penalty = enable_length_penalty

        self.temperature = kwargs.get("temperature", 0.7)
        self.with_scale = kwargs.get("with_scale", True)

        self._system_prompt_token_length_val = None

        self.iteration_counter = 0

        if self.exclude_system_prompt:
            print(
                f"HiddenRewardManager initialized with system prompt exclusion enabled"
            )
        else:
            print(
                f"HiddenRewardManager initialized with system prompt exclusion disabled"
            )

        if self.enable_clustering:
            print(
                f"Auto-grouped clustering reward enabled with weight: {self.clustering_weight}"
            )

        print(f"Cross-group reward enabled with weight: {self.cross_group_weight}")

        if self.enable_length_penalty:
            print(
                f"Length penalty enabled: -1 reward for max length outputs without EOS token"
            )

    def _find_subsequence(self, main_list: List[Any], sub_list: List[Any]) -> int:

        len_sub = len(sub_list)
        for i in range(len(main_list) - len_sub + 1):
            if main_list[i : i + len_sub] == sub_list:
                return i
        return -1

    def _get_system_prompt_token_length(self) -> int:

        if self._system_prompt_token_length_val is not None:
            return self._system_prompt_token_length_val

        if not self.exclude_system_prompt:
            self._system_prompt_token_length_val = 0
            return 0

        print("Calculating system prompt token length for pooling mask...")

        try:

            dummy_user_content = "some_unique_string_for_testing_user_content"

            messages = [
                {"role": "system", "content": self.custom_prompt},
                {"role": "user", "content": dummy_user_content},
            ]

            try:
                if (
                    hasattr(self.tokenizer, "chat_template")
                    and self.tokenizer.chat_template is not None
                ):
                    full_prompt_str = self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                else:

                    full_prompt_str = (
                        f"{self.custom_prompt}\n\n{dummy_user_content}\n\nAssistant:"
                    )
            except Exception as e:

                full_prompt_str = (
                    f"{self.custom_prompt}\n\n{dummy_user_content}\n\nAssistant:"
                )

            full_tokens = self.tokenizer.encode(
                full_prompt_str, add_special_tokens=False
            )

            user_content_tokens = self.tokenizer.encode(
                dummy_user_content, add_special_tokens=False
            )

            user_content_start_index = self._find_subsequence(
                full_tokens, user_content_tokens
            )

            if user_content_start_index == -1:

                self._system_prompt_token_length_val = 0
            else:

                self._system_prompt_token_length_val = user_content_start_index
                print(
                    f"✓ System prompt prefix length determined: {self._system_prompt_token_length_val} tokens. "
                    "These will be excluded from pooling."
                )

        except Exception as e:
            print(f"Error calculating system prompt length: {e}. Defaulting to 0.")
            self._system_prompt_token_length_val = 0

        return self._system_prompt_token_length_val

    def compute_similarity(self, hidden1, hidden2, method="cosine"):

        if method == "cosine":
            return F.cosine_similarity(hidden1, hidden2, dim=-1)
        elif method == "dot_product":
            return torch.sum(hidden1 * hidden2, dim=-1)
        else:
            raise ValueError("method must be 'cosine' or 'dot_product'")

    def extract_mean_pooled_hidden_states_batch(self, hidden_states, attention_masks):

        attention_masks_float = attention_masks.float()  # [batch_size, seq_len]

        pooling_mask = attention_masks_float.clone()

        if self.exclude_system_prompt:
            system_prompt_len = self._get_system_prompt_token_length()

            if system_prompt_len > 0:

                for k in range(pooling_mask.shape[0]):

                    valid_token_indices = torch.where(pooling_mask[k] == 1)[0]

                    if len(valid_token_indices) > 0:

                        valid_start_pos = valid_token_indices[0].item()
                        valid_end_pos = valid_token_indices[-1].item()

                        system_prompt_end_pos = valid_start_pos + system_prompt_len

                        if (
                            system_prompt_end_pos <= valid_end_pos + 1
                        ):  # +1 because end_pos is inclusive

                            pooling_mask[k, valid_start_pos:system_prompt_end_pos] = 0
                        else:

                            print(
                                f"Batch item {k}: system prompt length ({system_prompt_len}) exceeds valid token length ({len(valid_token_indices)}). Skipping exclusion."
                            )
                    else:
                        print(
                            f"Batch item {k}: no valid tokens found in attention mask."
                        )

        pooling_mask_expanded = pooling_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]

        masked_hidden_states = (
            hidden_states * pooling_mask_expanded
        )  # [batch_size, seq_len, hidden_dim]

        valid_lengths = pooling_mask.sum(dim=1, keepdim=True)  # [batch_size, 1]

        valid_lengths = torch.clamp(valid_lengths, min=1)

        sum_hidden_states = masked_hidden_states.sum(dim=1)  # [batch_size, hidden_dim]
        mean_pooled_hidden_states = (
            sum_hidden_states / valid_lengths
        )  # [batch_size, hidden_dim]

        mean_pooled_hidden_states = torch.nn.functional.normalize(
            mean_pooled_hidden_states, p=2, dim=1
        )

        return mean_pooled_hidden_states

    def _auto_group_by_prompts(self, prompts_tensor):

        batch_size = prompts_tensor.shape[0]
        group_ids = [-1] * batch_size
        group_dict = defaultdict(list)
        next_group_id = 0

        for i in range(batch_size):
            current_prompt = prompts_tensor[i]

            found_group = False
            for j in range(i):
                if torch.equal(current_prompt, prompts_tensor[j]):

                    group_ids[i] = group_ids[j]
                    group_dict[group_ids[i]].append(i)
                    found_group = True
                    break

            if not found_group:

                group_ids[i] = next_group_id
                group_dict[next_group_id].append(i)
                next_group_id += 1

        print(f"Auto-grouped {batch_size} samples into {len(group_dict)} groups")
        for group_id, indices in group_dict.items():
            print(f"Group {group_id}: {len(indices)} samples at indices {indices}")

        return group_ids, group_dict

    def _reorder_by_groups(self, query_hidden, main_hidden, neg_hidden, group_dict):

        reorder_indices = []
        group_sizes = []

        for group_id in sorted(group_dict.keys()):
            group_indices = group_dict[group_id]
            reorder_indices.extend(group_indices)
            group_sizes.append(len(group_indices))

        restore_indices = [0] * len(reorder_indices)
        for new_idx, original_idx in enumerate(reorder_indices):
            restore_indices[original_idx] = new_idx

        reordered_query = query_hidden[reorder_indices]
        reordered_main = main_hidden[reorder_indices]
        reordered_neg = neg_hidden[reorder_indices]

        print(f"Reordered data by groups: {group_sizes}")

        return (
            reordered_query,
            reordered_main,
            reordered_neg,
            reorder_indices,
            group_sizes,
            restore_indices,
        )

    def _compute_rewards_on_reordered_data(
        self,
        reordered_query,
        reordered_main,
        reordered_neg,
        group_sizes,
        similarity_method="cosine",
    ):

        batch_size = reordered_query.shape[0]
        device = reordered_query.device
        num_groups = len(group_sizes)

        positive_rewards = torch.zeros(batch_size, device=device)
        negative_rewards = torch.zeros(batch_size, device=device)
        clustering_rewards = torch.zeros(batch_size, device=device)
        cross_group_rewards = torch.zeros(batch_size, device=device)

        group_all_mains = []
        start_idx = 0
        for group_size in group_sizes:
            end_idx = start_idx + group_size

            group_all_mains.append(reordered_main[start_idx:end_idx])
            start_idx = end_idx

        start_idx = 0

        for group_idx, group_size in enumerate(group_sizes):
            end_idx = start_idx + group_size

            group_query = reordered_query[start_idx:end_idx]  # [group_size, hidden_dim]
            group_main = reordered_main[start_idx:end_idx]  # [group_size, hidden_dim]
            group_neg = reordered_neg[start_idx:end_idx]  # [group_size, hidden_dim]

            group_positive_sim = self.compute_similarity(
                group_query, group_main, similarity_method
            )
            positive_rewards[start_idx:end_idx] = group_positive_sim

            if group_size > 0:

                group_negative_sim = self.compute_similarity(
                    group_query[0:1], group_neg[0:1], similarity_method
                )[0]

                negative_rewards[start_idx:end_idx] = -group_negative_sim

            if self.enable_clustering and group_size > 1:

                if similarity_method == "cosine":
                    normalized_main = F.normalize(group_main, p=2, dim=-1)
                    similarity_matrix = torch.mm(normalized_main, normalized_main.t())
                elif similarity_method == "dot_product":
                    similarity_matrix = torch.mm(group_main, group_main.t())
                else:
                    raise ValueError(
                        "similarity_method must be 'cosine' or 'dot_product'"
                    )

                for i in range(group_size):

                    clustering_reward_sum = 0.0
                    for j in range(group_size):
                        if i != j:
                            clustering_reward_sum += similarity_matrix[i, j]

                    clustering_reward_avg = clustering_reward_sum / (group_size - 1)
                    clustering_rewards[start_idx + i] = clustering_reward_avg

            if num_groups > 1:

                for i in range(group_size):
                    current_query = group_query[i : i + 1]  # [1, hidden_dim]
                    max_sims = []

                    for other_group_idx, other_group_mains in enumerate(
                        group_all_mains
                    ):
                        if other_group_idx != group_idx:

                            group_sims = self.compute_similarity(
                                current_query.expand(
                                    other_group_mains.shape[0], -1
                                ),  # [other_group_size, hidden_dim]
                                other_group_mains,  # [other_group_size, hidden_dim]
                                similarity_method,
                            )

                            max_sim = torch.max(group_sims)
                            max_sims.append(max_sim)

                    if len(max_sims) > 0:
                        cross_group_reward_avg = -torch.stack(max_sims).mean()
                        cross_group_rewards[start_idx + i] = cross_group_reward_avg

            print(
                f"Group {group_idx} (size={group_size}): "
                f"pos_sim={group_positive_sim.mean().item():.4f}, "
                f"neg_sim={negative_rewards[start_idx].item():.4f}, "
                f"clustering={clustering_rewards[start_idx:end_idx].mean().item():.4f}, "
                f"cross_group={cross_group_rewards[start_idx:end_idx].mean().item():.4f}"
            )

            start_idx = end_idx

        total_rewards = positive_rewards + negative_rewards
        if self.enable_clustering:
            total_rewards += self.clustering_weight * clustering_rewards
        total_rewards += self.cross_group_weight * cross_group_rewards

        return (
            total_rewards,
            positive_rewards,
            negative_rewards,
            clustering_rewards,
            cross_group_rewards,
        )

    def _restore_original_order(self, reordered_rewards, restore_indices):

        original_order_rewards = torch.zeros_like(reordered_rewards)

        for original_idx, reordered_idx in enumerate(restore_indices):
            original_order_rewards[original_idx] = reordered_rewards[reordered_idx]

        return original_order_rewards

    def reward_function(
        self,
        query_mean_hidden_states,
        main_mean_hidden_states,
        neg_mean_hidden_states,
        group_dict=None,
        temperature=0.7,
        similarity_method="cosine",
    ):

        if group_dict is None:

            sim_positive = self.compute_similarity(
                query_mean_hidden_states, main_mean_hidden_states, similarity_method
            )
            sim_negative = self.compute_similarity(
                query_mean_hidden_states, neg_mean_hidden_states, similarity_method
            )
            positive_sim_rewards = sim_positive
            negative_sim_rewards = -sim_negative
            clustering_rewards = torch.zeros_like(sim_positive)
            cross_group_rewards = torch.zeros_like(sim_positive)
            total_rewards = (
                positive_sim_rewards
                + negative_sim_rewards
                + self.cross_group_weight * cross_group_rewards
            )
            return (
                total_rewards,
                positive_sim_rewards,
                negative_sim_rewards,
                clustering_rewards,
                cross_group_rewards,
            )

        (
            reordered_query,
            reordered_main,
            reordered_neg,
            reorder_indices,
            group_sizes,
            restore_indices,
        ) = self._reorder_by_groups(
            query_mean_hidden_states,
            main_mean_hidden_states,
            neg_mean_hidden_states,
            group_dict,
        )

        (
            reordered_total,
            reordered_positive,
            reordered_negative,
            reordered_clustering,
            reordered_cross_group,
        ) = self._compute_rewards_on_reordered_data(
            reordered_query,
            reordered_main,
            reordered_neg,
            group_sizes,
            similarity_method,
        )

        total_rewards = self._restore_original_order(reordered_total, restore_indices)
        positive_sim_rewards = self._restore_original_order(
            reordered_positive, restore_indices
        )
        negative_sim_rewards = self._restore_original_order(
            reordered_negative, restore_indices
        )
        clustering_rewards = self._restore_original_order(
            reordered_clustering, restore_indices
        )
        cross_group_rewards = self._restore_original_order(
            reordered_cross_group, restore_indices
        )

        return (
            total_rewards,
            positive_sim_rewards,
            negative_sim_rewards,
            clustering_rewards,
            cross_group_rewards,
        )

    def reward_function_with_scale(
        self,
        query_mean_hidden_states,
        main_mean_hidden_states,
        neg_mean_hidden_states,
        group_dict=None,
        temperature=0.7,
        similarity_method="cosine",
        use_temperature=True,
    ):

        (
            total_rewards,
            positive_sim_rewards,
            negative_sim_rewards,
            clustering_rewards,
            cross_group_rewards,
        ) = self.reward_function(
            query_mean_hidden_states,
            main_mean_hidden_states,
            neg_mean_hidden_states,
            group_dict,
            temperature,
            similarity_method,
        )

        if use_temperature:
            positive_sim_rewards = positive_sim_rewards / temperature
            negative_sim_rewards = negative_sim_rewards / temperature
            if self.enable_clustering:
                clustering_rewards = clustering_rewards / temperature
            cross_group_rewards = cross_group_rewards / temperature

            total_rewards = positive_sim_rewards + negative_sim_rewards
            if self.enable_clustering and group_dict is not None:
                total_rewards += self.clustering_weight * clustering_rewards
            total_rewards += self.cross_group_weight * cross_group_rewards

        return (
            total_rewards,
            positive_sim_rewards,
            negative_sim_rewards,
            clustering_rewards,
            cross_group_rewards,
        )

    def _should_apply_length_penalty(
        self, data_item, prompt_length, valid_response_length
    ):

        try:

            response_ids = data_item.batch["responses"]

            max_response_length = response_ids.shape[-1]
            reached_max_length = valid_response_length == max_response_length

            if not reached_max_length:
                return False

            last_token_id = response_ids[valid_response_length - 1].item()

            eos_token_id = self._get_eos_token_id()
            print(f"find last_token_id: {eos_token_id}")

            is_not_eos = last_token_id != eos_token_id

            print(f"apply length penalty: {is_not_eos}")
            if is_not_eos:
                return True
            else:
                return False

        except Exception as e:
            print(f"Error checking length penalty: {e}")
            return False

    def _get_eos_token_id(self):

        if (
            hasattr(self.tokenizer, "eos_token_id")
            and self.tokenizer.eos_token_id is not None
        ):
            return self.tokenizer.eos_token_id
        elif (
            hasattr(self.tokenizer, "eos_token")
            and self.tokenizer.eos_token is not None
        ):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        elif (
            hasattr(self.tokenizer, "special_tokens_map")
            and "eos_token" in self.tokenizer.special_tokens_map
        ):
            eos_token = self.tokenizer.special_tokens_map["eos_token"]
            return self.tokenizer.convert_tokens_to_ids(eos_token)
        elif (
            hasattr(self.tokenizer, "<|end_of_text|>")
            and self.tokenizer.pad_token_id is not None
        ):  # llama
            return self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        else:

            print("Warning: Could not find EOS token ID, using default value 2")
            return 2

    def normalize_rewards(self, rewards, method="z_score"):
        if method == "z_score":
            return (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        elif method == "min_max":
            min_r, max_r = rewards.min(), rewards.max()
            return (rewards - min_r) / (max_r - min_r + 1e-8)
        elif method == "none":
            return rewards
        else:
            raise ValueError("method must be 'z_score', 'min_max', or 'none'")

    def decode_batch_for_debug(self, data, max_examples=5):

        debug_info = []

        for i in range(min(len(data), max_examples)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            query_ids = data_item.batch.get("query_prompts", None)
            negative_doc_ids = data_item.batch.get("negative_doc_prompts", None)

            prompt_length = prompt_ids.shape[-1]
            query_length = query_ids.shape[-1] if query_ids is not None else 0
            negative_doc_length = (
                negative_doc_ids.shape[-1] if negative_doc_ids is not None else 0
            )

            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            if query_ids is not None:
                query_response_ids = data_item.batch["query_responses"]
                valid_query_response_length = data_item.batch["query_attention_mask"][
                    query_length:
                ].sum()
                valid_query_response_ids = query_response_ids[
                    :valid_query_response_length
                ]
            else:
                valid_query_response_ids = None

            if query_ids is not None:
                valid_query_length = data_item.batch["query_attention_mask"][
                    :query_length
                ].sum()
                valid_query_ids = query_ids[-valid_query_length:]
            else:
                valid_query_ids = None

            if negative_doc_ids is not None:
                valid_negative_doc_length = data_item.batch[
                    "negative_doc_attention_mask"
                ][:negative_doc_length].sum()
                valid_negative_doc_ids = negative_doc_ids[-valid_negative_doc_length:]
            else:
                valid_negative_doc_ids = None

            if negative_doc_ids is not None:
                negative_doc_response_ids = data_item.batch["negative_doc_responses"]
                valid_negative_doc_response_length = data_item.batch[
                    "negative_doc_attention_mask"
                ][negative_doc_length:].sum()
                valid_negative_doc_response_ids = negative_doc_response_ids[
                    :valid_negative_doc_response_length
                ]
            else:
                valid_negative_doc_response_ids = None

            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=True
            )
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )
            query_prompt_str = (
                self.tokenizer.decode(valid_query_ids, skip_special_tokens=True)
                if valid_query_ids is not None
                else ""
            )
            query_response_str = (
                self.tokenizer.decode(
                    valid_query_response_ids, skip_special_tokens=True
                )
                if valid_query_response_ids is not None
                else ""
            )
            negative_doc_prompt_str = (
                self.tokenizer.decode(valid_negative_doc_ids, skip_special_tokens=True)
                if valid_negative_doc_ids is not None
                else ""
            )
            negative_doc_response_str = (
                self.tokenizer.decode(
                    valid_negative_doc_response_ids, skip_special_tokens=True
                )
                if valid_negative_doc_response_ids is not None
                else ""
            )

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            debug_info.append(
                {
                    "index": i,
                    "data_source": data_source,
                    "query_prompt_str": query_prompt_str,
                    "query_response_str": query_response_str,
                    "prompt_str": prompt_str,
                    "response_str": response_str,
                    "negative_doc_prompt_str": negative_doc_prompt_str,
                    "negative_doc_response_str": negative_doc_response_str,
                }
            )

        return debug_info

    def __call__(self, data: DataProto, hidden_states=None, return_dict=False):

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        if hidden_states is None:
            batch_size = len(data)
            reward_tensor = torch.zeros_like(
                data.batch["responses"], dtype=torch.float32
            )
            reward_extra_info = defaultdict(list)
            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            return reward_tensor

        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        negative_sim_rewards = None

        main_hidden_states = hidden_states.batch[
            "main_hidden_states"
        ]  # [batch_size, seq_len, hidden_dim]
        query_hidden_states = hidden_states.batch[
            "query_hidden_states"
        ]  # [batch_size, seq_len, hidden_dim]
        neg_hidden_states = hidden_states.batch.get(
            "neg_hidden_states", None
        )  # optional

        main_attention_masks = torch.stack(
            [data[i].batch["attention_mask"] for i in range(batch_size)]
        )
        query_attention_masks = torch.stack(
            [data[i].batch["query_attention_mask"] for i in range(batch_size)]
        )
        neg_attention_masks = None
        if neg_hidden_states is not None:
            neg_attention_masks = torch.stack(
                [
                    data[i].batch["negative_doc_attention_mask"]
                    for i in range(batch_size)
                ]
            )

        group_ids = None
        group_dict = None
        if self.enable_clustering:
            try:

                prompts_tensor = torch.stack(
                    [data[i].batch["prompts"] for i in range(batch_size)]
                )
                group_ids, group_dict = self._auto_group_by_prompts(prompts_tensor)
                print(f"Auto-grouped into {len(group_dict)} groups")
            except Exception as e:
                print(
                    f"Failed to auto-group by prompts: {e}. Disabling clustering for this batch."
                )
                group_dict = None

        main_mean_hidden_states = self.extract_mean_pooled_hidden_states_batch(
            main_hidden_states, main_attention_masks
        )
        query_mean_hidden_states = self.extract_mean_pooled_hidden_states_batch(
            query_hidden_states, query_attention_masks
        )
        if neg_hidden_states is not None:
            neg_mean_hidden_states = self.extract_mean_pooled_hidden_states_batch(
                neg_hidden_states, neg_attention_masks
            )

        if neg_hidden_states is not None:
            if self.with_scale:
                (
                    total_rewards,
                    positive_sim_rewards,
                    negative_sim_rewards,
                    clustering_rewards,
                    cross_group_rewards,
                ) = self.reward_function_with_scale(
                    query_mean_hidden_states,
                    main_mean_hidden_states,
                    neg_mean_hidden_states,
                    group_dict=group_dict,
                    temperature=self.temperature,
                )
            else:
                (
                    total_rewards,
                    positive_sim_rewards,
                    negative_sim_rewards,
                    clustering_rewards,
                    cross_group_rewards,
                ) = self.reward_function(
                    query_mean_hidden_states,
                    main_mean_hidden_states,
                    neg_mean_hidden_states,
                    group_dict=group_dict,
                    temperature=self.temperature,
                )
        else:

            batch_size = query_mean_hidden_states.shape[0]
            device = query_mean_hidden_states.device

            positive_sim_rewards = self.compute_similarity(
                query_mean_hidden_states, main_mean_hidden_states, method="cosine"
            )
            clustering_rewards = torch.zeros(batch_size, device=device)
            cross_group_rewards = torch.zeros(batch_size, device=device)

            if group_dict is not None:

                group_representative_queries = {}
                for gid, idxs in group_dict.items():
                    if len(idxs) > 0:
                        group_representative_queries[gid] = query_mean_hidden_states[
                            idxs[0]
                        ]

                for gid, idxs in group_dict.items():
                    idx_tensor = torch.tensor(idxs, device=device, dtype=torch.long)
                    group_main = main_mean_hidden_states.index_select(0, idx_tensor)
                    group_query = query_mean_hidden_states.index_select(0, idx_tensor)

                    if self.enable_clustering and group_main.shape[0] > 1:
                        normalized_main = F.normalize(group_main, p=2, dim=-1)
                        sim_mat = torch.mm(normalized_main, normalized_main.t())

                        for i, sample_idx in enumerate(idxs):
                            clustering_sum = sim_mat[i].sum() - sim_mat[i, i]
                            clustering_rewards[sample_idx] = clustering_sum / (
                                group_main.shape[0] - 1
                            )

                    if len(group_representative_queries) > 1:
                        other_queries = [
                            v
                            for ogid, v in group_representative_queries.items()
                            if ogid != gid
                        ]
                        if len(other_queries) > 0:
                            other_q = torch.stack(
                                other_queries, dim=0
                            )  # [num_other, hidden]
                            for i, sample_idx in enumerate(idxs):
                                current_main = group_main[i : i + 1]  # [1, hidden]
                                sims = self.compute_similarity(
                                    current_main.expand(other_q.shape[0], -1),
                                    other_q,
                                    method="cosine",
                                )
                                cross_group_rewards[sample_idx] = sims.mean()

            if self.with_scale:
                positive_sim_rewards = positive_sim_rewards / self.temperature
                if self.enable_clustering:
                    clustering_rewards = clustering_rewards / self.temperature
                cross_group_rewards = cross_group_rewards / self.temperature

            total_rewards = positive_sim_rewards.clone()
            if self.enable_clustering and group_dict is not None:
                total_rewards = (
                    total_rewards + self.clustering_weight * clustering_rewards
                )
            if group_dict is not None and len(group_dict) > 1:
                total_rewards = (
                    total_rewards - self.cross_group_weight * cross_group_rewards
                )

        for i in range(batch_size):
            data_item = data[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()

            final_reward = total_rewards[i]
            if self.enable_length_penalty and self._should_apply_length_penalty(
                data_item, prompt_length, valid_response_length
            ):
                final_reward = torch.tensor(
                    -1.0, device=total_rewards.device, dtype=total_rewards.dtype
                )
                print(
                    f"Applied length penalty to sample {i}: reached max length without EOS"
                )

            reward_tensor[i, valid_response_length - 1] = final_reward

        reward_extra_info["positive_sim_rewards"] = positive_sim_rewards.cpu().tolist()
        if neg_hidden_states is not None:
            reward_extra_info["negative_sim_rewards"] = (
                negative_sim_rewards.cpu().tolist()
            )
        reward_extra_info["clustering_rewards"] = clustering_rewards.cpu().tolist()
        reward_extra_info["cross_group_rewards"] = cross_group_rewards.cpu().tolist()
        reward_extra_info["total_rewards"] = total_rewards.cpu().tolist()
        if group_ids is not None:
            reward_extra_info["group_ids"] = group_ids
            # Store num_groups as a list with length equal to batch_size so that
            # downstream splitting utilities (e.g., DataProto.chunk) can safely
            # split the non-tensor fields along the batch dimension.
            reward_extra_info["num_groups"] = [len(group_dict)] * batch_size

        if self.num_examine > 0:

            self.iteration_counter += 1

            sample_idx = (self.iteration_counter - 1) % batch_size

            debug_info = self.decode_batch_for_debug(data, max_examples=batch_size)

            if sample_idx < len(debug_info):
                info = debug_info[sample_idx]
                i = info["index"]

                print(f"=== Iteration {self.iteration_counter} - Sample {i} ===")
                print(f"[Group: {group_ids[i] if group_ids else 'N/A'}]")
                print(f"[Data Source: {info['data_source']}]")
                print(f"[Query Prompt]: {info['query_prompt_str']}")
                print(f"[Query Response]: {info['query_response_str']}")
                print(f"[Positive Doc Prompt]: {info['prompt_str']}")
                print(f"[Positive Doc Response]: {info['response_str']}")
                if neg_hidden_states is not None:
                    print(f"[Negative Doc Prompt]: {info['negative_doc_prompt_str']}")
                    print(
                        f"[Negative Doc Response]: {info['negative_doc_response_str']}"
                    )
                print(f"[Total Reward]: {total_rewards[i].item():.4f}")
                print(f"[Positive Sim Reward]: {positive_sim_rewards[i].item():.4f}")
                if negative_sim_rewards is not None:
                    print(
                        f"[Negative Sim Reward]: {negative_sim_rewards[i].item():.4f}"
                    )
                if self.enable_clustering:
                    print(f"[Clustering Reward]: {clustering_rewards[i].item():.4f}")
                    print(f"[Clustering Weight]: {self.clustering_weight}")
                print(f"[Cross Group Reward]: {cross_group_rewards[i].item():.4f}")
                print(f"[Cross Group Weight]: {self.cross_group_weight}")
                if self.exclude_system_prompt:
                    print(
                        f"[System Prompt Excluded]: {self._get_system_prompt_token_length()} tokens"
                    )
                if self.enable_length_penalty:
                    print(
                        f"[Length Penalty Enabled]: -1 reward for max length outputs without EOS token"
                    )
                print("=" * 80)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
