<div align="center">
  <h1>💨 Notus 7B v1</h1>
  <img src="https://github.com/argilla-io/notus/assets/36760800/d50bbae1-16ec-40c5-8254-5c4ea60435da" alt="A banner representing Notus, the wind god of the south, in a mythical and artistic style. The banner features a strong, swirling breeze, embodying the warm, wet character of the southern wind. Gracefully flowing across the scene are several paper planes, caught in the gentle yet powerful gusts of Notus. The background is a blend of warm colors, symbolizing the heat of the south, with hints of blue and green to represent the moisture carried by this wind. The overall atmosphere is one of dynamic movement and warmth."/>
</div>

---

Notus 7B v1 is a DPO fine-tuned version of Zephyr 7B Beta SFT fine-tuned on UltraFeedback, but using the average of the different criterias to binarize the data, instead of the critique score; so that the chosen response is based on the average rather than on the critique score. All the training code and configuration has been adapted / ported from [`huggingface/alignment-handbook`](https://github.com/huggingface/alignment-handbook).

## Contents

Here you will find the following directories and files:

* [`fine-tune/`](./fine-tune/): contains the fine-tuning scripts adapted from [`huggingface/alignment-handbook`](https://github.com/huggingface/alignment-handbook) to suit our specific use cases and needs.

* [`eval/`](./eval/): contains the evaluation instructions and results from the benchmarks [`EleutherAI/lm-eval-harness`](https://github.com/EleutherAI/lm-eval-harness) (from the `big-refactor` branch), `MT-Bench` in [`lm-sys/FastChat`](https://github.com/lm-sys/FastChat), and `AlpacaEval` in [`tatsu-lab/alpaca_eval`](https://github.com/tatsu-lab/alpaca_eval).

* [`alt/`](./alt/): contains some files that have been used for experimentation purposes, but are not needed / required in order to reproduce and / or understand the work done with Notus 7B v1. Disclaimer: expect those files to change, be messy, and not work as intended.

## Model Details

### Model Description

- **Developed by:** Argilla (based on HuggingFace H4 and MistralAI previous efforts and amazing work)
- **Shared by:** Argilla
- **Model type:** GPT-like 7B model DPO fine-tuned
- **Language(s) (NLP):** Mainly English
- **License:** MIT (same as Zephyr 7B-beta)
- **Finetuned from model:** [`alignment-handbook/zephyr-7b-sft-full`](https://huggingface.co/alignment-handbook/zephyr-7b-sft-full)

### Model Variants

- [`notus-7b-v1`](https://huggingface.co/argilla/notus-7b-v1): full DPO fine-tuning
- [`notus-7b-v1-lora`](https://huggingface.co/argilla/notus-7b-v1): DPO fine-tuning using LoRA

> [!NOTE]
> Even though we have the LoRA weights within the 🤗 Hub, most of the experimentation / evaluation has been done using `notus-7b-v1` to do a fair comparison with `zephyr-7b-beta`.

## Performance

### Chat Benchmarks

Table adapted from Zephyr-7b-β original table for [MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench) and [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) benchmarks. Notus stays on par with Zephyr on MT-Bench, while surpassing Zephyr, Claude 2, and Cohere Command on AlpacaEval. Making Notus the most-competitive 7B commercial model on AlpacaEval.

| Model | Size | Alignment | MT-Bench (score) | AlpacaEval (win rate %) |
|-------------|-----|----|---------------|--------------|
| MPT-Chat |  7B |dSFT |5.42| -|
| Xwin-LMv0.1 | 7B| dPPO| 6.19| 87.83|
| Mistral-Instructv0.1 | 7B|  - | 6.84 |-|
| Zephyr-7b-β | 7B | dDPO | **7.34** | 90.60 |
| **notus-7b-v1** | 7B | dDPO | 7.30 | **91.42** |
| GPT-3.5-turbo | - |RLHF |7.94 |89.37|
| Claude 2 |  - |RLHF |8.06| 91.36|
| Cohere Command |  - |RLHF |-| 90.62|
| GPT-4 |  -| RLHF |8.99| 95.28|
| Falcon-Instruct |  40B |dSFT |5.17 |45.71|
| Guanaco | 65B |  SFT |6.41| 71.80|
| Llama2-Chat |  70B |RLHF |6.86| 92.66|
| Vicuna v1.3 |  33B |dSFT |7.12 |88.99|
| WizardLM v1.0 |  70B |dSFT |7.71 |-|
| Xwin-LM v0.1 |   70B |dPPO |- |95.57|

## Academic Benchmarks

* Results from [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard):

  | Model | Average | ARC | HellaSwag | MMLU | TruthfulQA | Winogrande | GSM8K | DROP |
  |-------|---------|-----|-----------|------|------------|------------|-------|------|
  | [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) | 52.15 | 62.03 | 84.36 | 61.07 | **57.45** | 77.74 | 12.74 | **9.66** |
  | **[argilla/notus-7b-v1](https://huggingface.co/argilla/notus-7b-v1)** | **52.89** | **64.59** | **84.78** | **63.03** | 54.37 | **79.4** | **15.16** | 8.91 |

* Results when running the evaluation locally from the `big-refactor` branch in `lm-eval-harness`:

  | Model | Average ⬆️ | ARC (25-s) ⬆️ | HellaSwag (10-s) ⬆️ | MMLU (5-s) ⬆️ | TruthfulQA (MC2) (0-s) ⬇️ | Winogrande (5-s) ⬇️ | GSM8K (5-s) ⬆️ | DROP (3-s) ⬇️ |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  |[HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) | 52.15 | 62.03 | 84.36 | 61.07 | 57.45 | 77.74 | 12.74 | 9.66 |
  | **[argilla/notus-7b-v1](https://huggingface.co/argilla/notus-7b-v1)** | **54.09** | 64.25 | 84.90 | 61.69 | 52.77 | 74.51 | 39.5 | 0.98 |

  The results from Mistral and Zephyr models retrieved from https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard, which **may not be fair as they are using a different revision of `lm-eval-harness`**, so may be worth re-running the benchmarks locally for Zephyr 7B Beta for a fair comparison.

## Training Details

### Training Hardware

We used VMs from different cloud providers based on their availability, but most of the experiments have been run in a VM with 8 x A100 40GB hosted in Google Cloud Platform (GCP), while some others in a similar VM in Lambda Labs, and lastly some extra experiments in an 8 x A100 80GB VM in RunPod. Meaning everything's adapted to work within 8 x A100 40GB.

### Training Data

We used a a new curated version of [`openbmb/UltraFeedback`](https://huggingface.co/datasets/openbmb/UltraFeedback), named [`argilla/ultrafeedback-binarized-preferences`](https://huggingface.co/argilla/ultrafeedback-binarized-preferences).

### Training Metrics

We've tracked all our metrics with [Weights and Biases](https://wandb.ai) (❤️), even though those are already within the 🤗 Hub using TensorBoard. But the metrics below are from an internal Weights and Biases report we've created for this project.

#### `notus-7b-v1`

<div align="center">
  <img width="873" alt="image" src="https://github.com/argilla-io/notus/assets/36760800/87b3f32b-41ff-4283-840f-b8d193b0cb34"/>
</div>

#### `notus-7b-v1-lora`

<div align="center">
  <img width="872" alt="image" src="https://github.com/argilla-io/notus/assets/36760800/aa81e8c3-da90-4e4c-a378-3c9902b421af"/>
</div>

## Reproduce

In order to reproduce the results of Notus 7B v1, please check [`fine-tune/`](./fine-tune/) to see the SFT and DPO fine-tuning scripts adapted from [`huggingface/alignment-handbook`](https://github.com/huggingface/alignment-handbook) to suit our specific use cases and needs.
