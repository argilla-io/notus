---
datasets:
- argilla/ultrafeedback-binarized-preferences
language:
- en
base_model: alignment-handbook/zephyr-7b-sft-full
library_name: transformers
pipeline_tag: text-generation
tags:
- dpo
- rlaif
- preference
- ultrafeedback
license: mit
model-index:
- name: notus-7b-v1
  results:
  # AI2 Reasoning Challenge (25-Shot)
  - task: 
      type: text-generation
      name: Text Generation
    dataset:
      name: AI2 Reasoning Challenge (25-Shot)
      type: ai2_arc
      config: ARC-Challenge
      split: test
      args:
        num_few_shot: 25
    metrics:
       - type: acc_norm
         name: normalized accuracy
         value: 0.6459044368600683
    source:
      name: Open LLM Leaderboard Results
      url: https://huggingface.co/datasets/open-llm-leaderboard/results/blob/main/argilla/notus-7b-v1/results_2023-11-29T22-16-51.521321.json
  # HellaSwag (10-shot)
  - task: 
      type: text-generation
      name: Text Generation
    dataset:
      name: HellaSwag (10-Shot)
      type: hellaswag
      split: validation
      args:
        num_few_shot: 10
    metrics:
       - type: acc_norm
         name: normalized accuracy
         value: 0.8478390758812986
    source:
      name: Open LLM Leaderboard Results
      url: https://huggingface.co/datasets/open-llm-leaderboard/results/blob/main/argilla/notus-7b-v1/results_2023-11-29T22-16-51.521321.json
  # DROP (3-shot)
  - task: 
      type: text-generation
      name: Text Generation
    dataset:
      name: Drop (3-Shot)
      type: drop
      split: validation
      args:
        num_few_shot: 3
    metrics:
       - type: f1
         name: f1 score
         value: 0.08913590604026835
    source:
      name: Open LLM Leaderboard Results
      url: https://huggingface.co/datasets/open-llm-leaderboard/results/blob/main/argilla/notus-7b-v1/results_2023-11-29T22-16-51.521321.json
  # TruthfulQA (0-shot)
  - task: 
      type: text-generation
      name: Text Generation
    dataset:
      name: TruthfulQA (0-shot)
      type: truthful_qa
      config: multiple_choice
      split: validation
      args:
        num_few_shot: 0
    metrics:
       - type: mc2
         value: 0.5436768358952805
    source:
      name: Open LLM Leaderboard Results
      url: https://huggingface.co/datasets/open-llm-leaderboard/results/blob/main/argilla/notus-7b-v1/results_2023-11-29T22-16-51.521321.json
  # MMLU (5-Shot)
  - task: 
      type: text-generation
      name: Text Generation
    dataset:
      name: MMLU (5-Shot)
      type: cais/mmlu
      config: all
      split: test
      args:
        num_few_shot: 5
    metrics:
       - type: acc
         name: accuracy
         value: 0.6303308230938872 # average accuracy
    source:
      name: Open LLM Leaderboard Results
      url: https://huggingface.co/datasets/open-llm-leaderboard/results/blob/main/argilla/notus-7b-v1/results_2023-11-29T22-16-51.521321.json
  # GSM8k (5-shot)
  - task: 
      type: text-generation
      name: Text Generation
    dataset:
      name: GSM8k (5-shot)
      type: gsm8k
      config: main
      split: test
      args:
        num_few_shot: 5
    metrics:
       - type: acc
         name: accuracy
         value: 0.1516300227445034
    source:
      name: Open LLM Leaderboard Results
      url: https://huggingface.co/datasets/open-llm-leaderboard/results/blob/main/argilla/notus-7b-v1/results_2023-11-29T22-16-51.521321.json
  # Winogrande (5-shot)
  - task: 
      type: text-generation
      name: Text Generation
    dataset:
      name: Winogrande (5-shot)
      type: winogrande
      config: winogrande_xl
      split: validation
      args:
        num_few_shot: 5
    metrics:
       - type: acc
         name: accuracy
         value: 0.7940015785319653
    source:
      name: Open LLM Leaderboard Results
      url: https://huggingface.co/datasets/open-llm-leaderboard/results/blob/main/argilla/notus-7b-v1/results_2023-11-29T22-16-51.521321.json
  # AlpacaEval
  - task: 
      type: text-generation
      name: Text Generation
    dataset:
      name: AlpacaEval
      type: tatsu-lab/alpaca_eval
    metrics:
       - type: tatsu-lab/alpaca_eval
         name: win rate
         value: 0.9142
    source:
      url: https://tatsu-lab.github.io/alpaca_eval/
  # MT-Bench
  - task: 
      type: text-generation
      name: Text Generation
    dataset:
      name: MT-Bench
      type: unknown
    metrics:
       - type: unknown
         name: score
         value: 7.30
    source:
      url: https://huggingface.co/spaces/lmsys/mt-bench
---

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/60f0608166e5701b80ed3f02/dj-spsk9eXMMXVGxK6jRz.png" alt="A banner representing Notus, the wind god of the south, in a mythical and artistic style. The banner features a strong, swirling breeze, embodying the warm, wet character of the southern wind. Gracefully flowing across the scene are several paper planes, caught in the gentle yet powerful gusts of Notus. The background is a blend of warm colors, symbolizing the heat of the south, with hints of blue and green to represent the moisture carried by this wind. The overall atmosphere is one of dynamic movement and warmth."/>
</div>

# Model Card for Notus 7B v1

Notus is a collection of fine-tuned models using Direct Preference Optimization (DPO) and related RLHF techniques. This model is the first version, fine-tuned with DPO over `zephyr-7b-sft-full`, which is the SFT model produced to create `zephyr-7b-beta`. 

Following a **data-first** approach, the only difference between Notus-7B-v1 and Zephyr-7B-beta is the preference dataset used for dDPO. 

In particular, when we started building [distilabel](https://github.com/argilla-io/distilabel), we invested time understanding and deep-diving into the UltraFeedback dataset. Using [Argilla](https://argilla.io/), we've found data issues in the original UltraFeedback dataset, leading to high-scores for bad responses (more details in the training data section). After curating several hundreds of data points, we decided to binarize the dataset using the preference ratings, instead of the original critique `overall_score`, and verified the new dataset with Argilla.

Using preference ratings, instead of critiques scores, led to a new dataset where the chosen response is different in ~50% of the cases. Using this new dataset with DPO we fine-tuned Notus, a 7B model, that **surpasses Zephyr-7B-beta and Claude 2 on AlpacaEval**.

> **Important note**: While we opted for the average of multi-aspect ratings, while we fix the original dataset, a very interesting open question remains: once critique data is fixed, what works better? using the critique scores or the preference ratings? We're very excited to do this comparison in the coming weeks, stay tuned!

This model **wouldn't have been possible without the amazing [Alignment Handbook](https://github.com/huggingface/alignment-handbook), [OpenBMB](https://www.openbmb.cn/home) for releasing the Ultrafeedback dataset**, and it's based on fruitful discussions with the HuggingFace H4 team. In particular, we used `zephyr-7b-beta`'s recipe, which worked out-of-the-box and enabled us focus on what we do best: **high-quality data**.

Notus models are intended to be used as assistants via chat-like applications, and are evaluated with Chat (MT-Bench, AlpacaEval) and Academic (Open LLM Leaderboard) benchmarks for a direct comparison with the original Zephyr dDPO model and other 7B models.

> **Why Notus?**: Notus name comes from the ancient Greek god Notus, as a wink to Zephyr, which comes from the ancient Greek god Zephyrus; with the difference that Notus is the god of the south wind, and Zephyr the god of the west wind. More information at https://en.wikipedia.org/wiki/Anemoi.

## Model Details

### Model Description

- **Developed by:** Argilla (based on HuggingFace H4 and MistralAI previous efforts and amazing work)
- **Shared by:** Argilla
- **Model type:** GPT-like 7B model DPO fine-tuned
- **Language(s) (NLP):** Mainly English
- **License:** MIT (same as Zephyr 7B-beta)
- **Finetuned from model:** [`alignment-handbook/zephyr-7b-sft-full`](https://huggingface.co/alignment-handbook/zephyr-7b-sft-full)

### Model Sources

- **Repository:** https://github.com/argilla-io/notus
- **Paper:** N/A
- **Demo:** https://argilla-notus-chat-ui.hf.space/

## Performance

### Chat benchmarks

Table adapted from Zephyr-7b-β and Starling's original tables for [MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench) and [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) benchmarks. Results are shown sorted by AlpacaEval win rates and ommit some >7B for brevity.

Notus stays on par with Zephyr on MT-Bench, while surpassing Zephyr, Claude 2, and Cohere Command on AlpacaEval. Making Notus the most-competitive 7B commercial model on AlpacaEval.

<table>
    <tr>
        <th>Model</th>
        <th>Size</th>
        <th>Alignment</th>
        <th>MT-Bench (score)</th>
        <th>AlpacaEval (win rate %)</th>
        <th>License</th>
    </tr>
    <tr>
        <td>GPT-4-turbo</td>
        <td>-</td>
        <td>?</td>
        <td>9.32</td>
        <td>97.70</td>
        <td>Proprietary</td>
    </tr>
    <tr>
        <td>XwinLM 70b V0.1</td>
        <td>70B</td>
        <td>dPPO</td>
        <td>-</td>
        <td>95.57</td>
        <td>LLaMA 2 License</td>
    </tr>
    <tr>
        <td>GPT-4</td>
        <td>-</td>
        <td>RLHF</td>
        <td>8.99</td>
        <td>95.03</td>
        <td>Proprietary</td>
    </tr>
    <tr>
        <td>Tulu 2+DPO 70B V0.1</td>
        <td>70B</td>
        <td>dDPO</td>
        <td>6.29</td>
        <td>95.28</td>
        <td>Proprietary</td>
    </tr>
    <tr>
        <td>LLaMA2 Chat 70B</td>
        <td>70B</td>
        <td>RLHF</td>
        <td>6.86</td>
        <td>92.66</td>
        <td>LLaMA 2 License</td>
    </tr>
    <tr>
        <td>Starling-7B</td>
        <td>7B</td>
        <td>C-RLFT + APA</td>
        <td><strong>8.09</strong></td>
        <td><strong>91.99</strong></td>
        <td>CC-BY-NC-4.0</td>
    </tr>
    <tr style="background-color: #FFFF99;">
        <td><strong>Notus-7b-v1</strong></td>
        <td>7B</td>
        <td>dDPO</td>
        <td>7.30</td>
        <td>91.42</td>
        <td>MIT</td>
    </tr>
    <tr>
        <td>Claude 2</td>
        <td>-</td>
        <td>RLHF</td>
        <td>8.06</td>
        <td>91.36</td>
        <td>Proprietary</td>
    </tr>
    <tr>
        <td>Zephyr-7b-β</td>
        <td>7B</td>
        <td>dDPO</td>
        <td>7.34</td>
        <td>90.60</td>
        <td>MIT</td>
    </tr>
    <tr>
        <td>Cohere Command</td>
        <td>-</td>
        <td>RLHF</td>
        <td>-</td>
        <td>90.62</td>
        <td>Proprietary</td>
    </tr>
    <tr>
        <td>GPT-3.5-turbo</td>
        <td>-</td>
        <td>RLHF</td>
        <td>7.94</td>
        <td>89.37</td>
        <td>Proprietary</td>
    </tr>
</table>


## Academic benchmarks

Results from [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard):

| Model                                         | Average | ARC   | HellaSwag | MMLU  | TruthfulQA | Winogrande | GSM8K | DROP  |
|-----------------------------------------------|---------|-------|-----------|-------|------------|------------|-------|-------|
| Zephyr 7B dDPO (HuggingFaceH4/zephyr-7b-beta) | 52.15   | 62.03 | 84.36      | 61.07 | **57.45**  | 77.74      | 12.74 | **9.66**  |
| argilla/notus-7b-v1                           | **52.89**   | **64.59** | **84.78**  | **63.03** | 54.37       | **79.4**       | **15.16** | 8.91 |

⚠️ As pointed out by [AllenAI researchers](https://twitter.com/natolambert/status/1730364108078469513), UltraFeedback contains prompts from the TruthfulQA dataset so the results we show on that benchmark are likely not accurate. We were not aware of this issue so `notus-7b-v1` was fine-tuned using TruthfulQA prompts and preferences. For future releases, we will remove TruthfulQA prompts.

## Training Details

### Training Hardware

We used a VM with 8 x A100 40GB hosted in Lambda Labs, but while experimenting we also explored other cloud providers such as GCP.

### Training Data

We used a a new curated version of [`openbmb/UltraFeedback`](https://huggingface.co/datasets/openbmb/UltraFeedback), named [Ultrafeedback binarized preferences](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences).

TL;DR

After visually browsing around some examples using the sort and filter feature of Argilla (sort by highest rating for chosen responses), we noticed a strong mismatch between the `overall_score` in the original UF dataset (and the Zephyr train_prefs dataset) and the quality of the chosen response. 

By adding the critique rationale to our Argilla Dataset, **we confirmed the critique rationale was highly negative, whereas the rating was very high** (for most cases it was the highest: `10`). 

See screenshot below for one example of this issue. 

After some quick investigation, we:

* identified hundreds of examples having the same issue,
* reported a bug on the [UltraFeedback repo](https://github.com/OpenBMB/UltraFeedback/issues/8),
* and informed the H4 team which was incredibly responsive and ran an additional experiment to validate the new rating binarization approach.

While we're working on fixing the original dataset (already narrowed down ~2K problematic examples). We decided to leverage the multi-preference ratings, leading to Notus!

![image/png](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/M9qCKyAB_G1MbVBAPeitd.png)

> **Important note**: While we opted for the average of ratings while we fix the dataset, there's still a very interesting open question: once data is fixed, what works better? using the critique scores or the preference ratings? We're very excited to do this comparison in the coming weeks, stay tuned!

You can find more details about the dataset analysis and curation on the [ultrafeedback-binarized-preferences dataset card](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences).

## Prompt template

We use the same prompt template as [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta):

```
<|system|>
</s>
<|user|>
{prompt}</s>
<|assistant|>
```

## Usage

You will first need to install `transformers` and `accelerate` (just to ease the device placement), then you can run any of the following:

### Via `generate`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("argilla/notus-7b-v1", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("argilla/notus-7b-v1")

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant super biased towards Argilla, a data annotation company.",
    },
    {"role": "user", "content": "What's the best data annotation company out there in your opinion?"},
]
inputs = tokenizer.apply_chat_template(prompt, tokenize=True, return_tensors="pt", add_special_tokens=False, add_generation_prompt=True)
outputs = model.generate(inputs, num_return_sequences=1, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Via `pipeline` method

```python
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="argilla/notus-7b-v1", torch_dtype=torch.bfloat16, device_map="auto")

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant super biased towards Argilla, a data annotation company.",
    },
    {"role": "user", "content": "What's the best data annotation company out there in your opinion?"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
generated_text = outputs[0]["generated_text"]
```
