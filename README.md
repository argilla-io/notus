# 💨 Notus 7B: DPO fine-tune of Zephyr 7B SFT

🤗 HuggingFace Hub Collection at https://huggingface.co/collections/argilla/notus-7b-dpo-fine-tune-of-zephyr-7b-sft-655529d7c73cb6c830e9555a

<div align="center">
  <img width="702" alt="image" src="https://github.com/argilla-io/notus-7b-dpo/assets/36760800/49bddbd2-ecfc-46d6-8d1d-1cb760dfe08b">
</div>

💥 Chat with Notus at https://argilla-notus-chat-ui.hf.space/ (powered by [`huggingface/chat-ui`](https://github.com/huggingface/chat-ui))

<div align="center">
  <img width="1624" alt="image" src="https://github.com/argilla-io/notus-7b-dpo/assets/36760800/a950f7f2-74ea-4873-a314-3afd1d4d7ac8">
</div>

## Installation

This wouldn't have been possible without the amazing work from the HuggingFace H4 Team and [`huggingface/alignment-handbook`](https://github.com/huggingface/alignment-handbook)!

```bash
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .
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

## DPO Fine-Tuning

```shell
WANDB_ENTITY=argilla-io WANDB_PROJECT=notus-7b-dpo ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py train_configs/config.yaml --use_flash_attention_2
```

Alternatively, if you prefer to use LoRA, you can also run:

```shell
WANDB_ENTITY=argilla-io WANDB_PROJECT=notus-7b-dpo ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/multi_gpu.yaml scripts/run_dpo.py train_configs/config_lora.yaml --use_flash_attention_2
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
