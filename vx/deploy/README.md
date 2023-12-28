# Deploy Notux 8X7b v1 on RunPod using TGI container

1. Go to RunPod dashboard an create a network storage (Storage tab). Notux safetensors files are around ~100GB, so create a storage with at least 150GB of space. Region should be CA-MTL-1, as is the region with more GPUs available.
2. Create the cheapest pod that you can an mount the network storage created in `/data`. We will use this pod to download the safetensors files.
3. Connect to the pod using SSH and install `pip install huggingface-hub`, then execute:
  ```python
  >>> for i in range(1, 20):
  ...     hf_hub_download("argilla/notux-8x7b-v1", f"model-{i:05d}-of-00019.safetensors", cache_dir="/data")
  ```
4. Once downloaded, you can delete the pod.
5. Create the pod using the `deploy.py` script. You will need to set the `RUNPOD_API_KEY` environment variable before. The script will mount the network storage in `/data` (you can change the network storage id using the `network_volume_id` parameter).
