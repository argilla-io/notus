# Habana

This directory contains scripts and some other instructions that are useful when running on Intel Habana (Gaudi HPUs). 

## Setup a working environment

The recommended way to work with Gaudi HPUs is within a Docker container that comes with all the dependencies needed to
use the HPUs installed, as described in this guide: [Habana - Run using containers](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#run-using-containers). If the machine has been provided using Intel Developer Cloud (IDC), then the bare metal machine should come with the required kernel and drivers already installed, and the Docker engine should be configured to use the `habana` runtime.

Having that said and assuming that the machine with the Gaudis was provided by IDC, the recommended way to setup a working environment is to create a VSCode Dev Container using one of the Habana Docker images. The `devcontainer.json` provided in this directory can be used. After creating the container, it's recommended to:

1. Install the latest version of `optimun-habana` from the Git repository:
  ```sh
  pip install git+https://github.com/huggingface/optimum-habana.git
  ```

2. Optionally, install the DeepSpeed adapted version for Habana:
  ```sh
  pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.13.0
  ```

## Scripts

### `inference.py`

The `inference.py` script is a simple script that using one HPU allows to create text generations using a `AutoModelForCausalLM` given a dataset containing instructions. For more information about the script, run `python inference.py --help`.
