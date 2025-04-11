import torch
import numpy as np

from RewardingVisualDoubt import dataset
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
from typing import Callable, List, Optional, Union


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


#############################################################################################
# Inherit from trl. PPOTrainer to define a MultimodalPPOTrainer allowing for image input
#############################################################################################


class MultimodalPPOTrainer(trl.PPOTrainer):

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

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
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
        queries: torch.Tensor,
        responses: torch.Tensor,
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
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        # TODO
        # For each sequence, determine the range of logits which represent the image embeddings
        # Select just the most probable token for the image embedding tokens
        # Mask out the entire image region

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {
                key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()
            }
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            logits, _, values = model(**input_kwargs)
            print(model_inputs["input_ids"].shape)
            print(values.shape)

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            input_ids_without_image_token = replace_image_token_with_another_token(
                input_ids.clone().detach(), replacement_token_id=0
            )  # N
            logprobs = logprobs_from_logits(
                logits[:, :-1, :], input_ids_without_image_token[:, 1:]
            )  # N (modified)
            print(logprobs.shape)
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = len(query_batch[j]) - 1
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

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )
