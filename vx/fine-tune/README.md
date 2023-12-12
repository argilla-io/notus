
# Fine-tuning

## Installation

This wouldn't have been possible without the amazing work from the HuggingFace H4 Team and [`huggingface/alignment-handbook`](https://github.com/huggingface/alignment-handbook)!

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/huggingface/alignment-handbook.git
pip install transformers==4.36.0
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

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/deepspeed_zero3.yaml run_dpo.py configs/accelerate.yaml
```

And with `nohup` as:

```shell
ACCELERATE_LOG_LEVEL=info nohup accelerate launch --config_file configs/deepspeed_zero3.yaml run_dpo.py configs/accelerate.yaml > output.log 2>&1 &
```
