Follow the steps below to reproduce the results of Notus 7B v1.

### Installation

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

### SFT Fine-Tuning

```bash
DS_SKIP_CUDA_CHECK=1 WANDB_ENTITY=argilla-io WANDB_PROJECT=notus-7b-sft ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3_a100_80gb_sft.yaml scripts/run_sft.py train_configs/config_a100_80gb_sft.yaml
```

### DPO Fine-Tuning

```shell
DS_SKIP_CUDA_CHECK=1 WANDB_ENTITY=argilla-io WANDB_PROJECT=notus-7b-dpo ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py train_configs/config_a100_40gb.yaml
```

Alternatively, if you prefer to use LoRA, you can also run:

```shell
WANDB_ENTITY=argilla-io WANDB_PROJECT=notus-7b-dpo ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/multi_gpu.yaml scripts/run_dpo.py train_configs/config_a100_40gb_lora.yaml
```

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
