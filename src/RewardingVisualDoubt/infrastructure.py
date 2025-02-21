import torch


def make_ipython_reactive_to_changing_codebase():
    from IPython.core.getipython import get_ipython

    ipython_client = get_ipython()
    if ipython_client:
        ipython_client.run_line_magic(magic_name="load_ext", line="autoreload")
        ipython_client.run_line_magic(magic_name="autoreload", line="2")


def human_readable_size(size_in_bytes):
    """
    Converts a size in bytes to a human-readable string (KB, MB, GB, etc.).
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_in_bytes < 1024.0:
            break
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} {unit}"


def chech_cuda_availablity():
    print(torch.cuda.is_available())


def check_memory_occupation_by_torch_tensors():

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated()
        max_memory_allocated = torch.cuda.max_memory_allocated()

        print(f"Current GPU memory occupied by tensors: {human_readable_size(memory_allocated)}")
        print(
            f"Maximum GPU memory occupied by tensors: {human_readable_size(max_memory_allocated)}"
        )

    else:
        print("CUDA is not available.")


def empty_torch_cuda_cache():
    torch.cuda.empty_cache()


def check_total_gpu_memory():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()  # Get the current device index
        properties = torch.cuda.get_device_properties(device)
        total_memory = properties.total_memory
        print(f"Total GPU memory: {human_readable_size(total_memory)}")
    else:
        print("CUDA is not available.")
