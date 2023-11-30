## Configuration files

This directory contains the configuration files either ported and/or adapted from [`huggingface/alignment-handbook`](https://github.com/huggingface/alignment-handbook) to suit our specific use cases and needs.

You will find the following directories and files:

* `accelerate/`: contains the 🤗 `accelerate` configuration files to run the distributed training in multiple GPUs either using or not [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed), and adapted for both 8 x A100 40GB (more accesible, cheaper), and for 8 x A100 80GB (less accesible, more expensive, used by HuggingFace H4). The 🤗 `accelerate` configuration used for the full fine-tunes is the DeepSpeed ZeRO 3, while the one for the LoRA ones is `multi_gpu.yaml`, and will work equally on both VMs.

* `sft/`: contains the configuration used for running another SFT fine-tuning over the previously SFT fine-tuned version of Zephyr, but was just created for experimentation purposes, as the Notus 7B v1 model is just the DPO fine-tune over the SFT fine-tuned version of Zephyr 7B Beta. So on, this is an experimental configuration, and we've only tested for full SFT fine-tuning in 8 x A100 80GB VMs.

* `dpo/`: contains the main configuration used for the DPO fine-tune that resulted in Notus 7B v1, as it contains the DPO configuration for both the full fine-tune and the LoRA one. We've only tested this configuration, adapted for our use case, and tested / ran in both scenarios (8 x A100 40GB and 8 x A100 80GB VMs) for the full DPO fine-tuning, and only in 8 x A100 40GB VMs for the LoRA one (but should work equally in 8 x A100 80GB VMs).