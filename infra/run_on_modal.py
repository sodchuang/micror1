"""
launch_modal.py

This tool is designed to launch scripts using https://modal.com/

It sets up the necessary environment, including GPU resources and python dependencies,
and executes the specified training script remotely.

### Setup and Usage
```bash
pip install modal
modal setup  # authenticate with Modal
export HF_TOKEN="your_huggingface_token"  # if using a gated model such as llama3
modal run --detach infra/run_on_modal.py --command "torchrun --standalone --nproc_per_node=8 train.py --wandb_log=True"
```

For iterative development, consider using `modal.Volume` to cache the model between runs to
avoid redownloading the weights at the beginning of train.py
"""

import os

import modal
from modal import FilePatternMatcher, gpu

app = modal.App("microR1")
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .pip_install("huggingface_hub[hf_transfer]")  # enable hf_transfer for faster downloads
    .add_local_dir(".", "/root/", ignore=~FilePatternMatcher("**/*.py"))  # copy root of microR1 to /root/ (only python files)
)
if "HF_TOKEN" not in os.environ:
    print("HF_TOKEN not found in environment variables, using an empty token.")
hf_token_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
if "WANDB_API_KEY" not in os.environ:
    print("WANDB_API_KEY not found in environment variables, using an empty token.")
wandb_api_key_secret = modal.Secret.from_dict({"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})


@app.function(
    gpu=gpu.A100(count=8, size="80GB"),
    image=image,
    timeout=24 * 60 * 60,  # 1 day
    secrets=[hf_token_secret, wandb_api_key_secret],
    network_file_systems={  # add persistent storage for HF cache
        "/root/models": modal.NetworkFileSystem.from_name("hf-cache", create_if_missing=True)
    },
)
def run_command(command: str):
    """This function will be run remotely on the specified hardware."""
    import shlex
    import subprocess

    # configure HF cache directory, enable faster hf hub transfers
    os.environ["HF_HOME"] = "/root/models"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    print(f"Running command: {command}")
    args = shlex.split(command)
    subprocess.run(args, check=True, cwd="/root", env=os.environ.copy())


@app.local_entrypoint()
def main(command: str):
    """Run a command remotely on modal.
    ```bash
    export HF_TOKEN="your_huggingface_token"  # if using a gated model such as llama3
    modal run --detach infra/run_on_modal.py --command "torchrun --standalone --nproc_per_node=8 train.py"
    ```
    """
    run_command.remote(command=command)
