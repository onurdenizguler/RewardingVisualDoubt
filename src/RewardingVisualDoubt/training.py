import torch
import numpy as np

import trl
from trl.models import PreTrainedModelWrapper
from trl.core import (
    PPODecorators,
    logprobs_from_logits,
    WANDB_PADDING,
    stack_dicts,
    stats_to_np,
    convert_to_scalar,
)
import time
import math
import peft
from typing import List
import typing


IGNORE_INDEX = -100
LLAVA_IMAGE_TOKEN_INDEX = -200  # as defined by the llava repo
TOKEN_INDEX_OF_THE_WORD_IMAGE = (
    1967  # 1967 is the index of the image token in the tokenizer (the word image)
)


def remove_padding(tensor, pad_token) -> torch.Tensor:
    # TODO: better alternative is output = generation[(1 - mask).sum() :]  # remove padding
    start_idx = 0
    # while start_idx < len(tensor) and tensor[start_idx] == pad_token:
    #     start_idx += 1

    # Find the end index where padding starts again
    end_idx = len(tensor) - 1
    while end_idx >= 0 and tensor[end_idx] == pad_token:
        end_idx -= 1

    # Slice the tensor to remove padding, add 1 to end_idx to include the last non-pad token
    trimmed_tensor = tensor[start_idx : end_idx + 1]
    return trimmed_tensor


def remove_preciding_padding_from_batch_tensor(batch: torch.Tensor):
    trimmed_sequences = []
    for seq in batch:
        # Find the first occurrence of token `1`
        ones = (seq == 1).nonzero(as_tuple=True)[0]
        if len(ones) > 0:
            first_one_idx = ones[0].item()
            trimmed_seq = seq[first_one_idx:]
            trimmed_sequences.append(trimmed_seq)
        else:
            raise Exception("Error at remove_preciding_padding_from_batch_tensor")

    # If you want to pad back to the same length (optional):
    return trimmed_sequences


def remove_trailing_padding_from_prediction(
    prediction: torch.Tensor, pad_token_id: int | None
) -> list[torch.Tensor]:
    """
    Remove padding tokens from the end of the tensor
    args:
        tensor: torch.Tensor: a batch of generations by an LM with each generation having trailing padding tokens at the end
        pad_token: int
    """
    assert pad_token_id is not None, "pad_token_id must be provided"
    return [remove_padding(p, pad_token_id) for p in prediction]


def replace_image_token_with_another_token(
    prediction: torch.Tensor,
    image_token_id: int = LLAVA_IMAGE_TOKEN_INDEX,
    replacement_token_id: int = TOKEN_INDEX_OF_THE_WORD_IMAGE,
) -> torch.Tensor:
    # TODO: Consider adding the special image token to tokenizer for future editions
    prediction[prediction == image_token_id] = replacement_token_id
    return prediction


def replace_image_token_with_another_token_for_list_of_tensors(
    predictions: list[torch.Tensor],
    image_token_id: int = LLAVA_IMAGE_TOKEN_INDEX,
    replacement_token_id: int = TOKEN_INDEX_OF_THE_WORD_IMAGE,
) -> list[torch.Tensor]:
    return [
        replace_image_token_with_another_token(p, image_token_id, replacement_token_id)
        for p in predictions
    ]


def get_likeliest_token_from_logits(
    logits: torch.Tensor,
):
    """
    Get the most likely token from the logits
    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
    Returns:
        torch.Tensor of shape (batch_size, sequence_length)
    """
    # Get the most likely token from the logits
    return torch.argmax(logits, dim=-1)


#############################################################################################
# Custom logic to get the indexes of the embeddings representing the input images in a batch
# built upon llava-biovil codebase's prepare_inputs_labels_for_multimodal() in LlavaMetaModel
#############################################################################################

from LLAVA_Biovil.llava import LlavaLlamaForCausalLM


def get_llava_image_embedding_index_range_for_multimodal_batch_for_ppo(
    model: (
        LlavaLlamaForCausalLM | trl.AutoModelForCausalLMWithValueHead | trl.PreTrainedModelWrapper
    ),
    input_ids,
    attention_mask,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the indexes of the embeddings representing the input images in a batch
    Args:
        input_ids: torch.Tensor of shape (batch_size, sequence_length)
        attention_mask: torch.Tensor of shape (batch_size, sequence_length)
        images: torch.Tensor of shape (batch_size, rgb_dim, height, weight)
        typically (batch_size, 3, 448, 448) for Radialog images given the processing pipeline
    """
    if isinstance(model, trl.PreTrainedModelWrapper):
        model = typing.cast(trl.AutoModelForCausalLMWithValueHead, model)
    if not isinstance(model, LlavaLlamaForCausalLM):
        if isinstance(model, trl.AutoModelForCausalLMWithValueHead):
            model_ = model.pretrained_model
        if isinstance(model_, peft.PeftModelForCausalLM):
            model_ = model_.base_model.model
        assert isinstance(
            model_, LlavaLlamaForCausalLM
        ), "Model must be an instance of LlavaLlamaForCausalLM or a PeftModelForCausalLM"
    else:
        model_ = model
    model_ = typing.cast(LlavaLlamaForCausalLM, model_)

    image_features = model_.encode_images(images).to(model_.device)
    attention_mask = attention_mask.clone().detach().bool()

    # 1. Remove paddings using attention_mask
    input_ids = [
        cur_input_ids[cur_attention_mask]
        for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    ]

    # 2. Go over each sequence in the batch and create a list of input embeddings
    new_input_embeds_list_batch: list[torch.Tensor] = (
        []
    )  # The input embeddings with image embeddings inserted
    image_indicator_mask_list_batch: list[torch.Tensor] = (
        []
    )  # list of tensors indicating whether the index carries and image embedding
    expanded_input_ids_list_batch: list[torch.Tensor] = (
        []
    )  # The input ids expanded with <unk> token index (0) where an image embedding is inserted

    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):

        # 2.1 Determine num_images and the indices where image tokens occur
        num_images = (cur_input_ids == LLAVA_IMAGE_TOKEN_INDEX).sum()
        image_token_indices = (
            [-1]
            + torch.where(cur_input_ids == LLAVA_IMAGE_TOKEN_INDEX)[0].tolist()
            + [cur_input_ids.shape[0]]
        )

        # 2.2 Collect the token_ids in the splits without images into a list
        cur_input_ids_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(
                cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
            )
        split_sizes = [x.shape[0] for x in cur_input_ids_noim]

        # 2.3 Embed the text tokens using the model and split them into a tuple of splits
        cur_input_embeds = model_.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im: list[torch.Tensor] = torch.split(
            cur_input_embeds, split_sizes, dim=0
        )

        # 2.4 Insert the image embeddings in between the splits and collect indicators showing at which indexes the image embeddings occur
        cur_new_input_embeds = []
        cur_image_token_indicators_list = []
        cur_expanded_input_ids_list = []
        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_expanded_input_ids_list.append(cur_input_ids_noim[i])
            cur_image_token_indicators_list.append(
                torch.zeros(
                    split_sizes[i], dtype=attention_mask.dtype, device=attention_mask.device
                )
            )
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_expanded_input_ids_list.append(
                    torch.zeros(
                        cur_image_features.shape[0],
                        dtype=cur_input_ids.dtype,
                        device=cur_input_ids.device,
                    )
                )
                cur_image_token_indicators_list.append(
                    torch.ones(
                        cur_image_features.shape[0],
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                )
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_expanded_input_ids_tensor: torch.Tensor = torch.cat(cur_expanded_input_ids_list)
        cur_image_token_indicators_tensor: torch.Tensor = torch.cat(cur_image_token_indicators_list)

        new_input_embeds_list_batch.append(cur_new_input_embeds)
        expanded_input_ids_list_batch.append(cur_expanded_input_ids_tensor)
        image_indicator_mask_list_batch.append(cur_image_token_indicators_tensor)

    # 3. Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(model_.config, "tokenizer_model_max_length", None)
    if tokenizer_model_max_length:
        max_len_orig = max(x.shape[0] for x in new_input_embeds_list_batch)
        if max_len_orig > tokenizer_model_max_length:
            print(
                f"Truncating sequences of len {max_len_orig} to {tokenizer_model_max_length} to fit the model's input length"
            )
        new_input_embeds_list_batch = [
            x[:tokenizer_model_max_length] for x in new_input_embeds_list_batch
        ]
        expanded_input_ids_list_batch = [
            x[:tokenizer_model_max_length] for x in expanded_input_ids_list_batch
        ]
        image_indicator_mask_list_batch = [
            x[:tokenizer_model_max_length] for x in image_indicator_mask_list_batch
        ]

    # 4. Combine the list of input embeddings into a single tensor via padding
    max_len = max(x.shape[0] for x in new_input_embeds_list_batch)
    batch_size = len(new_input_embeds_list_batch)

    # 4.1 Pad the list of sequences to the same length and combine them into a single tensor
    new_input_embeds_list_batch_padded = []
    expanded_input_ids_list_batch_padded = []
    image_indicator_mask_list_batch_padded = []
    attention_mask = torch.zeros(
        (batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device
    )
    for i, cur_new_embed in enumerate(new_input_embeds_list_batch):
        cur_len = cur_new_embed.shape[0]
        assert (
            getattr(model.config, "tokenizer_padding_side", "right") == "left"
        )  # We pad from the left assuming a decoder-only model
        new_input_embeds_list_batch_padded.append(
            torch.cat(
                (
                    torch.zeros(
                        (max_len - cur_len, cur_new_embed.shape[1]),
                        dtype=cur_new_embed.dtype,
                        device=cur_new_embed.device,
                    ),
                    cur_new_embed,
                ),
                dim=0,
            )
        )
        expanded_input_ids_list_batch_padded.append(
            torch.cat(
                (
                    torch.zeros(
                        (max_len - cur_len),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    expanded_input_ids_list_batch[i],
                ),
                dim=0,
            )
        )
        image_indicator_mask_list_batch_padded.append(
            torch.cat(
                (
                    torch.zeros(
                        (max_len - cur_len),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    image_indicator_mask_list_batch[i],
                ),
                dim=0,
            )
        )
        if cur_len > 0:
            attention_mask[i, -cur_len:] = True

    # 4.2 Stack the list of tensors into single tensors
    final_input_embeds: torch.Tensor = torch.stack(new_input_embeds_list_batch_padded, dim=0)
    final_expanded_input_ids: torch.Tensor = torch.stack(
        expanded_input_ids_list_batch_padded, dim=0
    )
    final_image_indicator_mask: torch.Tensor = torch.stack(
        image_indicator_mask_list_batch_padded, dim=0
    )

    return final_input_embeds, attention_mask, final_image_indicator_mask, final_expanded_input_ids


#############################################################################################
# Inherit from trl. PPOTrainer to define a MultimodalPPOTrainer allowing for image input
#############################################################################################


class MultimodalPPOTrainer(trl.PPOTrainer):

    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
    ):
        raise NotImplementedError(
            "The step function is not implemented for MultimodalPPOTrainer. Use multimodal_step instead."
        )

    @PPODecorators.empty_cuda_cache()
    def multimodal_step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        images: torch.Tensor,  # N
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            images (`torch.Tensor`):
                A batch of tensors containing the images to be processed by the model of shape
                (`batch_size`, `image_dim1`, `image_dim2`, `image_dim3`)

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        # The modifications made on the original non-multimodal step function are marked with a # N comment

        bs = self.config.batch_size

        queries, responses, scores = self._step_safety_checker(bs, queries, responses, scores)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = torch.tensor(scores).mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)
        model_inputs["images"] = images  # N

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        with torch.no_grad():
            print("First pass for logprobs")
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model, queries, responses, model_inputs, return_logits=full_kl_penalty
            )

            # for when the model is a peft model
            if self.is_peft_model and hasattr(
                self.accelerator.unwrap_model(self.model).pretrained_model,
                "disable_adapter",
            ):
                with self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter():
                    ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                        self.model, queries, responses, model_inputs, return_logits=full_kl_penalty
                    )
            elif self.is_peft_model and not hasattr(self.model.pretrained_model, "disable_adapter"):
                raise ValueError(
                    "You are using a `peft` version that does not support `disable_adapter`. Please update your `peft` version to the latest version."
                )

            else:
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.ref_model, queries, responses, model_inputs, return_logits=full_kl_penalty
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            print("Compute rewards and advantages")
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                rewards, non_score_reward = self.compute_rewards(
                    scores, all_logprobs, ref_logprobs, masks
                )
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        train_stats = {}
        for _ in range(self.config.ppo_epochs):
            print("Start epoch level training")
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                print(
                    "Start backward batch level training for batch size:",
                    self.config.backward_batch_size,
                )
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(
                    0, self.config.backward_batch_size, self.config.mini_batch_size
                ):
                    print(
                        "Start mini batch level training for batch size:",
                        self.config.mini_batch_size,
                    )
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        forward_pass_output = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                        )  # N
                        logprobs: torch.FloatTensor = typing.cast(
                            torch.FloatTensor, forward_pass_output[0]
                        )  # N
                        logits: torch.FloatTensor = typing.cast(
                            torch.FloatTensor, forward_pass_output[1]
                        )  # N
                        vpreds: torch.FloatTensor = typing.cast(
                            torch.FloatTensor, forward_pass_output[2]
                        )  # N

                        # Due to llava's input preperation logic, the minibatch logprobs, values and masks are padded to the max length of the batch
                        # We need to remove the excess padding from the logprobs and logits to fit the actual padding required by the minibatch at hand
                        mini_batch_excess_padding_len = (
                            mini_batch_dict["logprobs"].shape[1] - logprobs.shape[1]
                        )  # N
                        aligned_mini_batch_dict = {
                            "logprobs": mini_batch_dict["logprobs"][
                                :, mini_batch_excess_padding_len:
                            ],
                            "values": mini_batch_dict["values"][:, mini_batch_excess_padding_len:],
                            "masks": mini_batch_dict["masks"][:, mini_batch_excess_padding_len:],
                            "advantages": mini_batch_dict["advantages"][
                                :, mini_batch_excess_padding_len:
                            ],
                            "returns": mini_batch_dict["returns"][
                                :, mini_batch_excess_padding_len:
                            ],
                        }  # N
                        train_stats = self.train_minibatch(
                            aligned_mini_batch_dict["logprobs"],
                            aligned_mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            aligned_mini_batch_dict["masks"],
                            aligned_mini_batch_dict["advantages"],
                            aligned_mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(
            train_stats["policy/advantages"]
        ).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(
            train_stats["policy/advantages"], WANDB_PADDING
        )
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    @PPODecorators.empty_cuda_cache()
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor | list[torch.Tensor] | list[torch.LongTensor],
        responses: torch.Tensor | list[torch.Tensor] | list[torch.LongTensor],
        model_inputs: dict,
        return_logits: bool = False,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
            return_logits (`bool`, *optional*, defaults to `False`):
                Whether to return all_logits. Set to `False` if logits are not needed to reduce memory consumption.
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        # The modifications made on the original non-multimodal batched_forward_pass function are marked with a # N comment
        assert (
            not self.is_encoder_decoder
        ), "Encoder-decoder models are not supported in this implementation"

        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {
                key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()
            }
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            logits, _, values = model(**input_kwargs)

            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            _, attention_mask, image_indicator_mask, input_ids = (
                get_llava_image_embedding_index_range_for_multimodal_batch_for_ppo(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=input_kwargs["images"],
                )
            )  # N

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                num_image_embedding_tokens_in_query = (
                    image_indicator_mask[j].nonzero(as_tuple=True)[0].shape[0]
                )
                start = len(query_batch[j]) + num_image_embedding_tokens_in_query - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0]
                end = start + len(response_batch[j])

                masks[j, :start] = 0
                masks[j, end:] = 0

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

            all_logits, all_values, all_logprobs, all_masks = self.pad_mini_batches(
                all_logits,
                all_values,
                all_logprobs,
                all_masks,
            )

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if all_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @classmethod
    def pad_mini_batches(
        cls,
        all_logits: list[torch.Tensor],
        all_values: list[torch.Tensor],
        all_logprobs: list[torch.Tensor],
        all_masks: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Pad the mini-batches to the maximum sequence length in the batch.
        Args:
            all_logits: list[Tensor] List of logits tensors each tensor representing a mini-batch.
            all_values: list[Tensor] of values tensors each tensor representing a mini-batch..
            all_logprobs: list[Tensor] of logprobs tensors each tensor representing a mini-batch..
            all_masks: list[Tensor] of masks tensors each tensor representing a mini-batch.
        Returns:
            List of padded mini-batch logits, values, logprobs, and masks tensors with each mini-batch padded to the maximum sequence length.
        """

        max_seq_len = max(
            t.shape[1] for t in all_values
        )  # values, logits, masks all share full length
        max_logprobs_seq_len = max_seq_len - 1  # logprobs has one less

        def pad_tensor_list(tensor_list, target_len, pad_fn, pad_value=0):
            padded = []
            for t in tensor_list:
                pad_len = target_len - t.shape[1]  # 0th dim is batch size, 1st dim is seq_len
                if pad_len == 0:
                    padded.append(t)
                else:
                    padded_tensor = pad_fn(t, pad_len, pad_value)
                    padded.append(padded_tensor)
            return padded

        # Padding functions
        def pad_logits(t, pad_len, pad_value):
            return torch.nn.functional.pad(
                t, (0, 0, pad_len, 0), value=pad_value
            )  # pad from the left along dim=1 (seq_len)

        def pad_2d(t, pad_len, pad_value):
            return torch.nn.functional.pad(
                t, (pad_len, 0), value=pad_value
            )  # pad from the left along dim=1 (seq_len)

        # Apply padding
        if all_logits:
            padded_logits = pad_tensor_list(all_logits, max_seq_len, pad_logits)
        else:
            padded_logits = all_logits
        padded_values = pad_tensor_list(all_values, max_seq_len, pad_2d)
        padded_masks = pad_tensor_list(all_masks, max_seq_len, pad_2d)
        padded_logprobs = pad_tensor_list(all_logprobs, max_logprobs_seq_len, pad_2d)

        return padded_logits, padded_values, padded_logprobs, padded_masks
