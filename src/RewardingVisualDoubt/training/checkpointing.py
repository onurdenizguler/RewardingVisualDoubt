import datetime
import os
import pathlib as path

import peft
import trl


def save_best_eval_lora_adapters_to_dir(
    model: peft.PeftModelForCausalLM,
    epoch: int,
    step: int,
    out_dir: str | path.Path,
    name_of_fine_tuning: str,
):
    save_dir = os.path.join(
        out_dir,
        name_of_fine_tuning,
        datetime.datetime.now().strftime("%Y-%m-%d"),
        "best_eval_model_epoch_{}_step_{}".format(epoch + 1, step + 1),
    )
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)


def save_best_eval_lora_adapters_and_value_head_to_dir(
    ppo_trainer: trl.PPOTrainer,
    epoch: int,
    step: int,
    out_dir: str | path.Path,
    name_of_fine_tuning: str,
):
    save_dir = os.path.join(
        out_dir,
        name_of_fine_tuning,
        datetime.datetime.now().strftime("%Y-%m-%d"),
        "best_eval_model_epoch_{}_step_{}".format(epoch + 1, step + 1),
    )
    os.makedirs(save_dir, exist_ok=True)
    ppo_trainer.save_pretrained(save_dir)
