import os
import runpod

runpod.api_key = os.getenv("RUNPOD_API_KEY")

info = runpod.create_pod(
    name="TGI argilla/notus-8x7b-v1 for Chat UI",
    image_name="ghcr.io/huggingface/text-generation-inference:latest",
    gpu_type_id="NVIDIA A40",
    support_public_ip=True,
    start_ssh=False,
    gpu_count=4,
    volume_in_gb=20,
    volume_mount_path="/data",
    container_disk_in_gb=20,
    docker_args="--model-id argilla/notux-8x7b-v1",
    ports="80/http",
    network_volume_id="vgbnwbyb61",
)

print(info)
