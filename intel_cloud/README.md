# Training in the Intel Cloud (Habana Gaudi 2)

```bash
pip install optimum[habana]
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.12.0
pip install intel-extension-for-transformers  # or pip install git+pip install git+https://github.com/intel/intel-extension-for-transformers.git@main
```

## DPO Fine-tuning

Ideally, we should be able to run something like:

```bash
WANDB_ENTITY=argilla-io WANDB_PROJECT=notus-7b-dpo-intel ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml intel_cloud/run_dpo.py train_configs/config_habana_gaudi2.yaml
```

## References

- https://github.com/intel/intel-extension-for-transformers/blob/7d276cd9fb1f0d18109bc96b795ebf2eeaea39b4/intel_extension_for_transformers/neural_chat/examples/finetuning/dpo_pipeline/dpo_clm.py
- https://github.com/intel/intel-extension-for-transformers/blob/7d276cd9fb1f0d18109bc96b795ebf2eeaea39b4/intel_extension_for_transformers/transformers/dpo_trainer.py#L307

