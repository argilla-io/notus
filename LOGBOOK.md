## 🧪 Experiments

### `notus-7b-dpo-lora`

Initially, while we were waiting for the A100s 80GB VRAM to come (yes, we're GPU-poor 😞), we decided to initially run a experiment fine-tuning `zephyr-7b-sft-full`, which is the SFT fine-tuned version of `mistral-7b-v0.1` with a curated version of the UltraChat dataset (200k rows), using DPO and LoRA.

For our experiment, we decided to use a modified version of the UltraFeedback dataset that instead of using the overall score for each response's critique, we compute the mean of preference rating scores (honesty, instruction-following, helpfulness, truthfulness) for each response. So on, the distribution is 0.0-5.0 instead of 0.0-10.0, as opposed to the pre-processed UltraFeedback dataset by the HuggingFace H4 Team.

Besides that, we slightly modified the `alignment-handbook/scripts/run_dpo.py` script not to rely on the pre-defined data-processing defined at `alignment-handbook/src/alignment/data.py`. Regarding the configuration used, we just re-used the configuration defined under `alignment-handbook/recipes/zephyr-7b-beta/dpo/config_lora.yaml`, but adding some extra args such as the `wandb` logging and also the mechanism to push to the HuggingFace Hub.

Weights and Biases run available at https://wandb.ai/argilla-io/notus-7b-dpo-lora/runs/yjjpz2h3
Model in the HuggingFace Hub available at https://huggingface.co/argilla/notus-7b-dpo-lora

Note that we also had to define a script named `upload.py` in order to merge and upload the merged model into the HuggingFace Hub instead of only the adapters, that you can also find at https://huggingface.co/argilla/notus-7b-dpo-lora-adapter

### `notus-7b-dpo`

Once we trained the LoRA adapters and tested the performance of it, we were happy with the results so we decided to run the full DPO fine-tuning instead. But as mentioned before, we (in Europe, or at least us) are GPU-poor, meaning that it was a hustle to be able to get some on demand VMs with 8 x A100 80GB VRAM and even the availability for 8 x A100 40GB VRAM was also lacking.

At first we decided to use GCP as our cloud provider which went great for some days and we were able to fine-tune the LoRA model on spot VMs, we could even get to use the 8 x A100 80GB thanks to the GCP person that contacted us to arrange everything, but again, just on spot, so we were able to run some experiments (expensive ones :() but the VMs kept on being preempted. So then we decided to give Lambda Labs a try and it was pretty easy and smooth to get the 8 x A100 40GB VMs up and running on demand.

Since the original configuration for the full DPO fine-tuning of `zephyr-7b-sft-full` (including the DeepSpeed ZeRO 3 configuration) was prepared for a VM with 8 x A100 80GB, we had to tweak some params to make it work on the 8 x A100 40GB we had. Below are the params we had to change, but long story short we had to load and train the model using a lower precision (BF16) and also making use of the Flash Attention 2 mechanism.

```diff
# Model arguments
model_name_or_path: alignment-handbook/zephyr-7b-sft-full
+ torch_dtype: auto
+ use_flash_attention_2: true

# Data training arguments
dataset_mixer:
  argilla/ultrafeedback-binarized-avg-rating-for-dpo: 1.0
dataset_splits:
\- train
preprocessing_num_workers: 12

# DPOTrainer arguments
bf16: true
beta: 0.1
do_eval: true
evaluation_strategy: steps
eval_steps: 100
gradient_accumulation_steps: 1
gradient_checkpointing: true
learning_rate: 5.0e-7
log_level: info
logging_steps: 10
lr_scheduler_type: linear
max_length: 1024
max_prompt_length: 512
num_train_epochs: 3
optim: rmsprop
output_dir: data/notus-7b-dpo
per_device_train_batch_size: 8
per_device_eval_batch_size: 4
push_to_hub: true
hub_model_id: argilla/notus-7b-dpo
hub_private_repo: true
save_strategy: epoch
save_total_limit: 1
seed: 42
warmup_ratio: 0.1
report_to:
  - wandb
  - tensorboard

```

And w.r.t. the DeepSpeed ZeRO 3 configuration, we had to also enable BF16 fine-tuning and setting the CPU as the offload device for the optimizer and the parameters, and also enabling the pin memory for the parameters.

```diff
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
+  bf16_enabled: true
  deepspeed_multinode_launcher: standard
+  offload_optimizer_device: cpu 
+  offload_param_device: cpu
+  offload_param_pin_memory: true
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

With those configuration changes, we were able to successfully run the full DPO fine-tuning in 8 x A100 40GB in around 12 hours. See more information about the run at https://wandb.ai/argilla-io/notus-7b-dpo/runs/p27wz8ix, and the model in the HuggingFace Hub at https://huggingface.co/argilla/notus-7b-dpo.
