import logging
import os
import subprocess

import psutil
import requests

from .shared import DEFAULT_RADLLAMA_GGUF_DIR, Quantization, create_radllama_model_filename

LLAMA_SERVER_BIN = "/home/guests/deniz_gueler/repos/llama.cpp/build/bin/llama-server"
PORT = 8080
N_GPU_LAYERS = 999
HEALTH_ENDPOINT = f"http://localhost:{PORT}/health"

DEFAULT_CTX_SIZE = 4096
DEFAULT_PARALLEL_QUERIES = 5


def _get_cpu_threads() -> int:
    slurm_threads = os.environ.get("SLURM_CPUS_ON_NODE", None)
    return int(slurm_threads) if slurm_threads else 8  # conservative fallback


def is_server_alive() -> bool:
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=2)
        return r.status_code == 200
    except:
        return False


def terminate_llama_server(server: subprocess.Popen):
    server.terminate()


def start_llama_server(
    port: int = PORT,
    gguf_dir: str = DEFAULT_RADLLAMA_GGUF_DIR,
    quantization: Quantization = Quantization.Q4_K_M,
    ctx_size: int = DEFAULT_CTX_SIZE,
) -> subprocess.Popen:
    num_threads = str(_get_cpu_threads())
    logging.info(f"Starting llama-server with {num_threads} threads")
    return subprocess.Popen(
        [
            LLAMA_SERVER_BIN,
            "--model",
            gguf_dir + "/" + create_radllama_model_filename(quantization),
            "--n-gpu-layers",
            str(N_GPU_LAYERS),
            "--ctx-size",
            str(ctx_size),
            "--cont-batching",
            "--threads",
            num_threads,
            "--port",
            str(port),
            # "--parallel",
            # str(3),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def kill_all_llama_servers():
    killed = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"]
            if cmdline and "llama-server" in cmdline[0]:
                logging.warning(f"Killing leftover llama-server (PID {proc.pid})")
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if killed == 0:
        logging.info("No llama-server processes found.")
    else:
        logging.info(f"✔️ Killed {killed} llama-server process(es).")
