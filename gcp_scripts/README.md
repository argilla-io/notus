## GCP

This directory contains a script used to allocate a VM on Google Compute Engine with 8 x A100 80GB VRAM, that runs forever until it's actually created since the demand in Europe is high and the availability is low, so we had to develop this script to actually be able to get the VM up and running, even though we eventually decided to move our experiments to Lambda Labs.

To run the script `europe_is_gpu_poor.sh` you just need to run the following command:
```bash
./europe_is_gpu_poor.sh <vm-name> <project-id> <zone> <service-account> <boot-disk-size>
```
