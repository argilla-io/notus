##!/bin/bash

# Check if required arguments are provided
if [ "$#" -ne 5 ]; then
	echo "Usage: $0 <vm-name> <project-id> <zone> <service-account> <boot-disk-size>"
	exit 1
fi

# Set GCP variables from command-line arguments
VM_NAME="$1"
PROJECT_ID="$2"
ZONE="$3"
SERVICE_ACCOUNT="$4"
BOOT_DISK_SIZE="$5"

# Delay between retry attempts (in seconds)
DELAY_SECONDS=120

# Variable to store the error message
ERROR_MESSAGE=""

# Function to create VM and capture error message
create_vm() {
	# Run gcloud command and capture error message
	ERROR_MESSAGE=$(gcloud compute instances create $VM_NAME \
		--project=$PROJECT_ID \
		--zone=$ZONE \
		--machine-type=a2-ultragpu-8g \
		--network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
		--maintenance-policy=TERMINATE \
		--provisioning-model=STANDARD \
		--service-account=$SERVICE_ACCOUNT \
		--scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
		--accelerator=count=8,type=nvidia-a100-80gb \
		--create-disk=auto-delete=yes,boot=yes,device-name=$VM_NAME,image=projects/ml-images/global/images/c0-deeplearning-common-gpu-v20231105-debian-11-py310,mode=rw,size=$BOOT_DISK_SIZE,type=projects/$PROJECT_ID/zones/$ZONE/diskTypes/pd-balanced \
		--no-shielded-secure-boot \
		--shielded-vtpm \
		--shielded-integrity-monitoring \
		--labels=goog-ec-src=vm_add-gcloud \
		--reservation-affinity=any 2>&1 >/dev/null)
}

# Function to parse error code from error message
parse_error_code() {
	local code=$(echo "$1" | awk '/^code:/ {print $2}')
	echo "$code"
}

# Main loop to create VM
attempt=1
while true; do
	echo "[ATTEMPT $attempt] Creating VM (8 x A100 80 GB) in $ZONE..."
	create_vm

	# Check if the VM creation was successful
	if [ $? -eq 0 ]; then
		echo "[ATTEMPT $attempt] VM created successfully!"
		break
	else
		error_code=$(parse_error_code "$ERROR_MESSAGE")
		echo "[ATTEMPT $attempt] VM creation failed. Retrying in $DELAY_SECONDS seconds..."
		echo "[ATTEMPT $attempt] Error code: $error_code"
		sleep $DELAY_SECONDS
		((attempt++))
	fi
done
