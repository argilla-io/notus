# Criticon

This folder contains some tests for the initial labeller.

1) start a standard pod in runpod (put enough container and disk volume)
2) create a venv
3) install torch
4) then follow the instructions of installation from axolotl.

When creating a pod with the axolotl image it asks for a password to connect via ssh, which I don't now where to get...

```bash
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

pip install packaging, wheel
pip install -e '.[flash-attn,deepspeed]'
```

Specific version of flash attention that worked for me
```bash
pip install flash-attn==2.3.1.post1
```

Before starting running your scripts, open python and check the following
```python
import torch
print(torch.version)
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
```

Prepare sample dataset
```bash
python sample_ds.py
```

Preprocess (more to test the dataset works)
```bash
python -m axolotl.cli.preprocess configs/criticon_lora.yml
```

Test run

```bash
accelerate launch -m axolotl.cli.train configs/criticon_lora.yml
```

To get the model merged (has to be pushed afterwards).

```bash
python -m axolotl.cli.merge_lora configs/criticon_lora.yml
```

To do some inference in the terminal

```bash
python -m axolotl.cli.inference configs/criticon_lora.yml --lora_model_dir="./criticon_v0"
```
