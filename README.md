# 💨 Notus 7B: DPO fine-tune of Zephyr 7B SFT

## Full training examples (DPO-only)

```shell
WANDB_ENTITY=argilla-io WANDB_PROJECT=notus-7b-dpo-full ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml recipes/notus-7b/dpo/modified_run_dpo.py recipes/notus-7b/dpo/config_full.yaml
```

## LoRA training examples

```shell
WANDB_ENTITY=argilla-io WANDB_PROJECT=notus-7b-dpo-lora ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml recipes/notus-7b/dpo/modified_run_dpo.py recipes/notus-7b/dpo/config_lora.yaml 
```
