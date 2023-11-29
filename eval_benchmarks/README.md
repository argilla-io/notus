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

The machine that we had at that time was one from Intel Developer Cloud (IDC) with HPUs, so we generated the outputs using the script `habana/scripts/inference.py` from this repo:

```sh
python inference.py --model_name_or_path argilla/notus-7b-v1 --dataset_name tatsu-lab/alpaca_eval --dataset_subset alpaca_eval --dataset_split "eval[:400]" --dataset_column instruction --max_new_tokens 2048 --output_file notus-7b-v1-1.jsonl
```

```sh
python inference.py --model_name_or_path argilla/notus-7b-v1 --dataset_name tatsu-lab/alpaca_eval --dataset_subset alpaca_eval --dataset_split "eval[400:]" --dataset_column instruction --max_new_tokens 2048 --output_file notus-7b-v1-2.jsonl
```

We executed the script twice, one for the first 400 examples and another one for the rest of the examples in order to avoid memory issues. Then we merged the outputs in a single file and executed the `alpaca_eval`:

```sh
export OPENAI_API_KEY=<YOUR_API_KEY>
pip install alpaca-eval
alpaca_eval evaluate --model_outputs notus-7b-v1.json
```

### Results

Results can be found in `alpaca_eval/annotation_alpaca_eval_gpt4.json` and `leaderboard.csv`:

```diff
                       win_rate  standard_error  n_total  avg_length
gpt4_turbo                97.70            0.51      804        2049
gpt4                      95.28            0.72      805        1365
llama-2-70b-chat-hf       92.66            0.91      804        1790
+ notus-7b-v1               91.42            0.99      804        2139
claude-2                  91.36            0.99      804        1069
cohere                    90.62            1.02      805        1983
chatgpt                   89.37            1.08      804         827
claude                    88.39            1.11      805        1082
llama-2-13b-chat-hf       81.09            1.38      804        1513
wizardlm-13b              75.31            1.51      804         985
guanaco-65b               71.80            1.59      805        1249
llama-2-7b-chat-hf        71.37            1.59      805        1479
vicuna-13b                70.43            1.61      805        1037
oasst-rlhf-llama-33b      66.52            1.66      805        1079
text_davinci_003          50.00            0.00      805         307
falcon-40b-instruct       45.71            1.75      805         662
alpaca-farm-ppo-human     41.24            1.73      805         803
alpaca-7b                 26.46            1.54      805         396
text_davinci_001          15.17            1.24      804         296
```


## MT-Bench (WIP)

...
