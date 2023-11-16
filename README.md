# 💨 Notus 7B: DPO fine-tune of Zephyr 7B SFT

Introduction (WIP)

## Installation (WIP)
```shell
...

> **Warning**
> When trying to run the scripts mentioned above and also the ones defined in the original `alignment-handbook`, we found out that the `bitsandbytes` dependency was running into some issues with the environment variable `GOOGLE_VM_CONFIG_LOCK_FILE`, so if you are running on GCP you should edit the `bitsandbytes/cuda_setup/env_vars.py` file to include the environment variable within the `to_be_ignored` function.

  ```diff
  def to_be_ignored(env_var: str, value: str) -> bool:
      ignorable = {
          "PWD",  # PWD: this is how the shell keeps track of the current working dir
          "OLDPWD",
          "SSH_AUTH_SOCK",  # SSH stuff, therefore unrelated
          "SSH_TTY",
          "HOME",  # Linux shell default
          "TMUX",  # Terminal Multiplexer
          "XDG_DATA_DIRS",  # XDG: Desktop environment stuff
          "HOME",  # Linux shell default
          "TMUX",  # Terminal Multiplexer
          "XDG_DATA_DIRS",  # XDG: Desktop environment stuff
          "XDG_GREETER_DATA_DIR",  # XDG: Desktop environment stuff
          "XDG_RUNTIME_DIR",
          "MAIL",  # something related to emails
          "SHELL",  # binary for currently invoked shell
          "DBUS_SESSION_BUS_ADDRESS",  # hardware related
          "PATH",  # this is for finding binaries, not libraries
          "LESSOPEN",  # related to the `less` command
          "LESSCLOSE",
  +       "GOOGLE_VM_CONFIG_LOCK_FILE", #avoids issues with Permissions on GCP, covered in- https://github.com/TimDettmers/bitsandbytes/issues/620#issuecomment-1666014197
          "_",  # current Python interpreter
      }
      return env_var in ignorable
  ```

  More information at https://github.com/TimDettmers/bitsandbytes/issues/620

---

## Full training examples (DPO-only)

```shell
WANDB_ENTITY=argilla-io WANDB_PROJECT=notus-7b-dpo-full ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml recipes/notus-7b/dpo/modified_run_dpo.py recipes/notus-7b/dpo/config_full.yaml
```

## LoRA training examples

```shell
WANDB_ENTITY=argilla-io WANDB_PROJECT=notus-7b-dpo-lora ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml recipes/notus-7b/dpo/modified_run_dpo.py recipes/notus-7b/dpo/config_lora.yaml 
```
