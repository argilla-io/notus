# Evaluation Benchmarks

## LM Evaluation Harness from EleutherAI 

### Install

Install `EleutherAI/lm-eval-harness` as follows from the `big-refactor` branch (still unstable):

`pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@big-refactor`

Then we run it as it follows using `accelerate` (we ran it in a VM with 4 x A100 40GB):

```bash
accelerate launch -m lm_eval --model hf --model_args pretrained=argilla/notus-7b-dpo,dtype=bfloat16 --tasks <TASK> --batch_size <BATCH_SIZE> --num_fewshot <NUM_FEW_SHOT> --output_path <OUTPUT_PATH>
```

> [!NOTE]
> We ran the experiments in a VM with 4 x A100 40GB in GCP, so if you're another configuration, less or more VRAM, etc. please make sure to tweak the arguments used within the `lm-eval` command. Find more details at https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor.

Or without using `accelerate` as:

```bash
lm_eval --model hf --model_args pretrained=argilla/notus-7b-dpo,dtype=bfloat16 --tasks <TASK> --batch_size <BATCH_SIZE> --num_fewshot <NUM_FEW_SHOT> --output_path <OUTPUT_PATH>
```

### Results

| Model | Average ⬆️ | ARC (25-s) ⬆️ | HellaSwag (10-s) ⬆️ | MMLU (5-s) ⬆️ | TruthfulQA (MC2) (0-s) ⬇️ | Winogrande (5-s) ⬇️ | GSM8K (5-s) ⬆️ | DROP (3-s) ⬇️ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|[mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) | 50.32 | 59.58 | 83.31 | 64.16 | 42.15 | 78.37 | 18.12 | 6.14 |
|[HuggingFaceH4/zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) | 52.4 | 61.01 | 84.04 | 61.39 | 57.9 | 78.61 | 14.03 | 9.82 |
|[HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) | 52.15 | 62.03 | 84.36 | 61.07 | 57.45 | 77.74 | 12.74 | 9.66 |
| **Ours** | **54.09** | 64.25 | 84.90 | 61.69 | 52.77 | 74.51 | 39.5 | 0.98 |

Results from Mistral and Zephyr models retrieved from https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

### Reproduce

* **ARC**
    ```bash
    accelerate launch -m lm_eval --model hf --model_args pretrained=argilla/notus-7b-dpo,dtype=bfloat16 --tasks arc_challenge --batch_size 8 --num_fewshot 25 --output_path arc_results
    ```
* **HellaSwag**:
    ```bash
    accelerate launch -m lm_eval --model hf --model_args pretrained=argilla/notus-7b-dpo,dtype=bfloat16 --tasks hellaswag --batch_size 8 --num_fewshot 10 --output_path hellaswag_results
    ```
* **MMLU**
    ```bash
    accelerate launch -m lm_eval --model hf --model_args pretrained=argilla/notus-7b-dpo,dtype=bfloat16 --tasks mmlu --batch_size 4 --num_fewshot 5 --output_path mmlu_results
    ```
* **TruthfulQA**
    ```bash
    accelerate launch -m lm_eval --model hf --model_args pretrained=argilla/notus-7b-dpo,dtype=bfloat16 --tasks truthfulqa --batch_size 8 --num_fewshot 0 --output_path truthfulqa_results
    ```
* **Winogrande**
    ```bash
    accelerate launch -m lm_eval --model hf --model_args pretrained=argilla/notus-7b-dpo,dtype=bfloat16 --tasks winogrande --batch_size 8 --num_fewshot 5 --output_path winogrande_results
    ```
* **GSM8K**
    ```bash
    accelerate launch -m lm_eval --model hf --model_args pretrained=argilla/notus-7b-dpo,dtype=bfloat16 --tasks gsm8k --batch_size 8 --num_fewshot 5 --output_path gsm8k_results
    ```
* **DROP**
    ```bash
    accelerate launch -m lm_eval --model hf --model_args pretrained=argilla/notus-7b-dpo,dtype=bfloat16 --tasks drop --batch_size 2 --num_fewshot 3 --output_path drop_results
    ```

## AlpacaEval

### Install

Install the AlpacaEval Python package as follows:

```bash
pip install alpaca-eval
```

Then set the environment variable with your OpenAI API key:

```bash
export OPENAI_API_KEY="<OPENAI_API_KEY>"
```

And finally, we prepare the configuration file to use to run the evaluation using AlpacaEval: 

```yaml
zephyr-7b-beta:
  prompt_template: "notus-prompt.txt"
  fn_completions: "huggingface_local_completions"
  completions_kwargs:
    model_name: "argilla/notus-7b-v1"
    model_kwargs:
      torch_dtype: "bfloat16"
    max_new_tokens: 2048
    temperature: 0.7
    top_p: 1.0
    do_sample: True
  pretty_name: "Notus 7B v1"
  link: "https://huggingface.co/argilla/notus-7b-v1"
```

Finally, we run it as:

```bash
alpaca_eval evaluate_from_model alpaca_eval-config/notus-7b-v1.yaml
```

## MT-Bench (WIP)

...
