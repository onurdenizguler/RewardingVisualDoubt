import warnings


def supress_known_warnings():
    warnings.filterwarnings(
        "ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0"
    )


def make_ipython_reactive_to_changing_codebase():
    from IPython.core.getipython import get_ipython

    ipython_client = get_ipython()
    if ipython_client:
        ipython_client.run_line_magic(magic_name="load_ext", line="autoreload")
        ipython_client.run_line_magic(magic_name="autoreload", line="2")
