## 🧪 Experiments

### `noble-wave-1`

Initially, while we were waiting for the A100s 80GB VRAM to come (yes, we're GPU-poor 😞), we decided to initially run a experiment fine-tuning `zephyr-7b-sft-full`, which is the SFT fine-tuned version of `mistral-7b-v0.1` with a curated version of the UltraChat dataset (200k rows), using DPO and LoRA.

For our experiment, we decided to use a modified version of the UltraFeedback dataset that instead of using the overall score for each response's critique, we compute the mean of preference rating scores (honesty, instruction-following, helpfulness, truthfulness) for each response. So on, the distribution is 0.0-5.0 instead of 0.0-10.0, as opposed to the pre-processed UltraFeedback dataset by the HuggingFace H4 Team.

Besides that, we slightly modified the `alignment-handbook/scripts/run_dpo.py` script not to rely on the pre-defined data-processing defined at `alignment-handbook/src/alignment/data.py`. Regarding the configuration used, we just re-used the configuration defined under `alignment-handbook/recipes/zephyr-7b-beta/dpo/config_lora.yaml`, but adding some extra args such as the `wandb` logging and also the mechanism to push to the HuggingFace Hub.

Weights and Biases run available at https://wandb.ai/argilla-io/notus-7b-dpo-lora/runs/yjjpz2h3
Model in the HuggingFace Hub available at https://huggingface.co/argilla/notus-7b-dpo-lora

Note that we also had to define a script named `upload.py` in order to merge and upload the merged model into the HuggingFace Hub instead of only the adapters, that you can also find at https://huggingface.co/argilla/notus-7b-dpo-lora-adapter

