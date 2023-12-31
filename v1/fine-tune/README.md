# Fine-tuning

Here you will find the following directories and files:

* [`configs/`](configs/): contains the configuration files either ported and/or adapted from [`huggingface/alignment-handbook`](https://github.com/huggingface/alignment-handbook) to suit our specific use cases and needs.

* [`run_dpo.py`](run_dpo.py): contains the main script to run the DPO fine-tuning, and is adapted from [`huggingface/alignment-handbook`](https://github.com/huggingface/alignment-handbook) to suit our specific use cases and needs. This is the main script that we used to fine-tune Notus 7B v1, and the file you should use to fine-tune your own models.

* [`run_sft.py`](run_sft.py): contains the main script to run the SFT fine-tuning, and is adapted from [`huggingface/alignment-handbook`](https://github.com/huggingface/alignment-handbook) to suit our specific use cases and needs. Note that this was only an attempt of SFT fine-tuning the previous SFT fine-tuned dataset of Zephyr 7B Beta, and was just created for experimentation purposes, as the Notus 7B v1 model is just the DPO fine-tune over the SFT fine-tuned version of Zephyr 7B Beta.

## Installation

This wouldn't have been possible without the amazing work from the HuggingFace H4 Team and [`huggingface/alignment-handbook`](https://github.com/huggingface/alignment-handbook)!

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/huggingface/alignment-handbook.git
```

Finally, if you are willing to push your models to the HuggingFace Hub, you should also login first via
`huggingface-cli login` and then install Git-LFS as `sudo apt-get install git-lfs`.

> [!TIP]
> Additionally, installing both `flash-attn` and `wandb` is recommended. `flash-attn` for a more
> efficient usage of the VRAM thanks to the Flash Attention 2 mechanism which also implies and speed-up; and
> `wandb` to also keep track of the experiments on Weights and Biases <3.
> ```bash
> python -m pip install flash-attn --no-build-isolation
> python -m pip install wandb
> ```
> If you installed `wandb` above you should also login via `wandb login`

### Installation on GCP

> [!WARNING]
> When trying to run the scripts mentioned above and also the ones defined in the original `alignment-handbook`, we found out that the `bitsandbytes` dependency was running into some issues with the environment variable `GOOGLE_VM_CONFIG_LOCK_FILE`, so if you are running on GCP you should edit the `bitsandbytes/cuda_setup/env_vars.py` file to include the environment variable within the `to_be_ignored` function.
> ```diff
> def to_be_ignored(env_var: str, value: str) -> bool:
>      ignorable = {
>          "PWD",  # PWD: this is how the shell keeps track of the current working dir
>          "OLDPWD",
>          "SSH_AUTH_SOCK",  # SSH stuff, therefore unrelated
>          "SSH_TTY",
>          "HOME",  # Linux shell default
>          "TMUX",  # Terminal Multiplexer
>          "XDG_DATA_DIRS",  # XDG: Desktop environment stuff
>          "HOME",  # Linux shell default
>          "TMUX",  # Terminal Multiplexer
>          "XDG_DATA_DIRS",  # XDG: Desktop environment stuff
>          "XDG_GREETER_DATA_DIR",  # XDG: Desktop environment stuff
>          "XDG_RUNTIME_DIR",
>          "MAIL",  # something related to emails
>          "SHELL",  # binary for currently invoked shell
>          "DBUS_SESSION_BUS_ADDRESS",  # hardware related
>          "PATH",  # this is for finding binaries, not libraries
>          "LESSOPEN",  # related to the `less` command
>          "LESSCLOSE",
>  +       "GOOGLE_VM_CONFIG_LOCK_FILE", #avoids issues with Permissions on GCP, covered in- https://github.com/TimDettmers/bitsandbytes/issues/620#issuecomment-1666014197
>          "_",  # current Python interpreter
>      }
>      return env_var in ignorable
> ```
> More information at https://github.com/TimDettmers/bitsandbytes/issues/620

## DPO Fine-tuning

To reproduce the DPO full fine-tuning, you can run the following command (assuming you are running it in a VM with 8 x A100 40GB GPUs, see [`configs/`](configs/) for more information on the different configuration files):

```shell
WANDB_ENTITY=<YOUR_WANDB_ENTITY> WANDB_PROJECT=<YOUR_WANDB_PROJECT> ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/accelerate/a100_40gb/deepspeed_zero3.yaml run_dpo.py configs/dpo/full/a100_40gb.yaml
```

Alternatively, if you prefer to use LoRA, you can also run:

```shell
WANDB_ENTITY=<YOUR_WANDB_ENTITY> WANDB_PROJECT=<YOUR_WANDB_PROJECT> ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/accelerate/multi_gpu.yaml run_dpo.py configs/dpo/lora/a100_40gb.yaml
```

> [!TIP]
> If the `torch` version that you have installed is not compiled with CUDA 11.8, you may need to set the `DS_SKIP_CUDA_CHECK=1` environment variable, so that DeepSpeed doesn't complain about the CUDA version; but note that it may also cause some issues and it is not recommended, so please make sure the `torch` version you have installed is compiled with the required CUDA version.
