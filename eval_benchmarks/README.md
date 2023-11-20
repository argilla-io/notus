## Evaluation Benchmarks

### MMLU

#### Install

Install `EleutherAI/lm-eval-harness` as follows from the `big-refactor` branch (still unstable):

`pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@big-refactor`

Then we run it as it follows using `accelerate` (we ran it in a VM with 4 x A100 40GB):

```bash
accelerate launch -m lm_eval --model hf --model_args pretrained=argilla/notus-7b-dpo,dtype=bfloat16 --tasks mmlu --batch_size 4 --num_fewshot 5 --output_path eval_results
```

> [!NOTE]
> We ran the experiments in a VM with 4 x A100 40GB in GCP, so if you're another configuration, less or more VRAM, etc. please make sure to tweak the arguments used within the `lm-eval` command. Find more details at https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor.

Or without using `accelerate` as:

```bash
lm_eval --model hf --model_args pretrained=argilla/notus-7b-dpo,dtype=bfloat16 --tasks mmlu --batch_size 4 --num_fewshot 5 --output_path eval_results
```

#### Results

Model | # Params | Dtype | Accuracy | # Few-shot |
------|----------|-------|----------|------------|
`argilla/notus-7b-dpo` | 7B | bfloat16 | 0.6169 | 5 |
`HuggingFaceH4/zephyr-7b-alpha` | 7B | bfloat16 | 0.6139 | 5 |
`HuggingFaceH4/zephyr-7b-beta` | 7B | bfloat16 | 0.6104 | 5 |

Results from Zephyr models retrieved from https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard