
# Fine-tuning

## Installation

This wouldn't have been possible without the amazing work from the HuggingFace H4 Team and [`huggingface/alignment-handbook`](https://github.com/huggingface/alignment-handbook)!

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.0
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

## DPO Fine-tuning

To reproduce the DPO full fine-tuning, you can run the following command (assuming you are running it in a VM with 8 x A100 40GB GPUs, see [`configs/`](configs/) for more information on the different configuration files):

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/accelerate/a100_80gb/deepspeed_zero3.yaml run_dpo.py configs/dpo/full/a100_80gb.yaml
```

And with `nohup` as:

```shell
ACCELERATE_LOG_LEVEL=info nohup accelerate launch --config_file configs/accelerate/a100_80gb/deepspeed_zero3.yaml run_dpo.py configs/dpo/full/a100_80gb.yaml > output.log 2>&1 &
```
