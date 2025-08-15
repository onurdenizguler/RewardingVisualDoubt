import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Iterator
import dataclasses
import random
import os
import pickle
import subprocess
import hashlib
import sys
from threading import Thread
import math

from RewardingVisualDoubt import training, reward
from radialog_report_generation_ppo_training import train


TRIAL_DIR = Path(
    "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/logs/radialog_report_generation_ppo_training/trials"
)


def _default_results_dict(objective_key: str, maximise: bool) -> dict[str, Any]:
    return {
        "objective_key": objective_key,
        "maximise": maximise,
        "best_reward_config": None,
        "best_score": float("-inf") if maximise else float("inf"),
        "runs": [],
    }


def init_results_file(out_file: Path, objective_key: str, maximise: bool) -> dict[str, Any]:
    if out_file.exists():
        return json.loads(out_file.read_text())
    data = _default_results_dict(objective_key, maximise)
    out_file.write_text(json.dumps(data, indent=2))
    return data


def persist(data: dict[str, Any], out_file: Path) -> None:
    tmp = out_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(out_file)


def build_reward_grid(base_cfg: reward.RewardConfig) -> Iterator[reward.RewardConfig]:

    # 42 options in total

    scalings = ["tanh", "logistic", "shifted", "centered"]
    NEXT_SCALINGS = ["tanh", "logistic", "shifted"]
    scale_options = [2.0, 5.0, 8.0]
    NEXT_SCALE_OPTIONS = [1.0, 1.5, 2.0]
    eps_options = [1e-3, 1e-2, 5e-2]
    NEXT_EPS_OPTIONS = [1e-2, 5e-2]
    squash_options = [math.log(1.2), math.log(1.3), math.log(1.4)]
    NEXT_SQUASH_OPTIONS = [math.log(1.5), math.log(1.8), math.log(2.0)]

    for scaling in scalings:
        for scale in scale_options:
            if scaling in {"tanh", "logistic"}:
                for squash_scale in squash_options:
                    yield dataclasses.replace(
                        base_cfg,
                        scaling=scaling,
                        scale=scale,
                        squash_scale=squash_scale,
                        eps=1e-3,
                        # keep eps unchanged
                    )
            else:
                for eps in eps_options:
                    yield dataclasses.replace(
                        base_cfg,
                        scaling=scaling,
                        scale=scale,
                        eps=eps,
                        squash_scale=None,
                    )


def objective_value(
    metrics: training.ReportGenerationRunFinalMetrics, key: str, maximise: bool
) -> float:
    val = getattr(metrics, key)
    return val if maximise else -val


def _final_metrics_path(
    metaparameters: training.TrainingMetaParameters,
    hyperparameters: training.ReportGenerationPPOHyperparameters,
) -> Path:
    return (
        TRIAL_DIR
        / "final_metrics"
        / f"final_metrics_{training.create_hash_out_of_parameters(metaparameters, hyperparameters)}.pkl"
    )


def dump_trial_objects(
    metaparameters: training.TrainingMetaParameters,
    hyperparameters: training.ReportGenerationPPOHyperparameters,
) -> str:
    name = f"trial_{training.create_hash_out_of_parameters(metaparameters, hyperparameters)}.pkl"
    path = TRIAL_DIR / name
    with open(path, "wb") as f:
        pickle.dump(
            {"metaparameters": metaparameters, "hyperparameters": hyperparameters},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return str(path)


def run_and_tee_pty(cmd, check=True, cwd=None, env=None):
    import pty, select, time

    """
    Runs `cmd` under a pseudo-TTY so tools like tqdm render progress bars.
    Mirrors live output to this process's stdout (captured by SLURM) and
    returns the combined output as `stdout`. `stderr` is empty because
    PTYs merge streams.
    """
    master_fd, slave_fd = pty.openpty()

    # Ensure unbuffered Python child (good for real-time logs)
    env = dict(os.environ if env is None else env)
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        cmd,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        cwd=cwd,
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)

    # Non-blocking reads
    try:
        os.set_blocking(master_fd, False)
    except AttributeError:
        pass  # Py<3.5 fallback not needed on most systems

    chunks = []
    try:
        # Read until process exits and PTY drains
        while True:
            # poll for readability with a small timeout
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in r:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    data = b""
                if data:
                    # Mirror to SLURM log
                    sys.stdout.buffer.write(data)
                    sys.stdout.flush()
                    chunks.append(data)
                else:
                    # EOF from PTY
                    break
            # When proc done, keep draining a bit and then exit
            if proc.poll() is not None and not r:
                # give a brief chance for trailing bytes to arrive
                time.sleep(0.05)
                # try one last read
                try:
                    tail = os.read(master_fd, 4096)
                except OSError:
                    tail = b""
                if tail:
                    sys.stdout.buffer.write(tail)
                    sys.stdout.flush()
                    chunks.append(tail)
                break

        ret = proc.wait()
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass

    combined = b"".join(chunks).decode(errors="replace")
    if check and ret != 0:
        # stderr is merged into stdout under a PTY; we return it in output
        raise subprocess.CalledProcessError(ret, cmd, output=combined, stderr="")
    return ret, combined, ""  # (stdout, stderr merged)


def run_trial_pickle(
    metaparameters: training.TrainingMetaParameters,
    hyperparameters: training.ReportGenerationPPOHyperparameters,
) -> training.ReportGenerationRunFinalMetrics:
    spec_path = dump_trial_objects(metaparameters, hyperparameters)

    cmd = [
        sys.executable,
        "-u",
        "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/workflows/report_generation/radialog_report_generation_ppo_training.py",
        "--spec",
        spec_path,
    ]
    retcode, stdout, stderr = run_and_tee_pty(cmd)

    final_metrics_path = _final_metrics_path(metaparameters, hyperparameters)
    with open(final_metrics_path, "rb") as f:
        metrics_obj: training.ReportGenerationRunFinalMetrics = pickle.load(f)
    return metrics_obj


def tune(objective_key: str, maximise: bool = True, sample: int | None = None) -> None:

    REWARD_TUNING_REPORT_GENERATION_PPO_TRAINING_CONFIGS_PATH = "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/workflows/report_generation/config_ppo_reward_tuning.yaml"
    meta_params, base_hparams = training.load_default_configs(
        REWARD_TUNING_REPORT_GENERATION_PPO_TRAINING_CONFIGS_PATH
    )
    grid = list(build_reward_grid(base_hparams.reward_config))

    if sample is not None and sample < len(grid):
        random.shuffle(grid)
        grid = grid[:sample]

    print("Grid:", grid)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(__file__).resolve().parent / f"reward_tuning_results_{ts}.json"
    results = init_results_file(results_path, objective_key, maximise)

    best_score: float = results["best_score"]
    best_cfg: reward.RewardConfig | None = (
        reward.RewardConfig(**results["best_reward_config"])
        if results["best_reward_config"]
        else None
    )

    done_keys = {json.dumps(rc, sort_keys=True) for rc in results["runs"]}

    for _, reward_cfg in enumerate(grid, start=1):
        key = json.dumps(dataclasses.asdict(reward_cfg), sort_keys=True)
        if key in done_keys:
            print(f"[Skip] Already evaluated: {reward_cfg}")
            continue

        print(
            f"[Run {len(results['runs']) + 1:03}] {reward_cfg.scaling=} {reward_cfg.eps=} "
            f"{reward_cfg.squash_scale=} {reward_cfg.scale=}"
        )

        hparams = dataclasses.replace(base_hparams, reward_config=reward_cfg)

        metrics = run_trial_pickle(meta_params, hparams)

        score = objective_value(metrics, objective_key, maximise)

        run_rec = {
            "reward_config": dataclasses.asdict(reward_cfg),
            "metrics": dataclasses.asdict(metrics),
            "objective": score,
        }
        results["runs"].append(run_rec)

        is_better = score > best_score if maximise else score < best_score
        if is_better:
            best_score, best_cfg = score, reward_cfg
            results["best_score"] = best_score
            results["best_reward_config"] = dataclasses.asdict(best_cfg)
            print(f" → new best: {best_score:.4f}")

        # Persist after each run
        persist(results, results_path)

    print("\nSearch complete ✅")
    print(f"Final results at: {results_path}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Grid-search reward_config using dataclasses (scale included)."
    )
    p.add_argument(
        "--objective-key",
        default="best_ece_and_conf_distribution_kl_eval",
        help="Metric name to optimise (attribute of ReportGenerationRunFinalMetrics).",
    )
    p.add_argument("--minimise", action="store_true", help="Lower is better (e.g. for ECE).")
    args = p.parse_args()

    tune(args.objective_key, maximise=not args.minimise, sample=10)

# Start with python /home/guests/deniz_gueler/repos/RewardingVisualDoubt/workflows/report_generation/radialog_report_generation_reward_tuning.py
