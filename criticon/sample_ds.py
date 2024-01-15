"""Script to prepare a sample dataset for SFT training with axolotl.

!pip install -q datasets transformers sentencepiece
"""

from datasets import load_dataset
from typing import Dict, Any
from transformers import PreTrainedTokenizer, AutoTokenizer

dataset_name = "argilla/ultrafeedback-critique"
local_path = "uf-critique/uf-critique.jsonl"
model_name = "teknium/OpenHermes-2.5-Mistral-7B"

print("Loading dataset")
uf_critique = load_dataset(dataset_name, split="train[:1000]")

print("Applying prompt template")

system_prompt = "User: A one-turn chat between a curious user and an artificial intelligence critique assistant."

notus_critique_instruction_template = """You are a critical teacher that provides specific, concise and constructive feedback for me in plain language, avoid giving me the reference response.

Consider how good the response follows the instruction:

<instruction>{instruction}</instruction>
<response>{response}</response>

Your answer must be in the following format:

<score>[1-10]</score>
<critique>your critique</critique>
"""

score_given_template = """<score>{score}</score>
<critique>{critique}</critique>
"""


def apply_chat_template_and_prepare_for_sft(
    example: Dict[str, Any], tokenizer: PreTrainedTokenizer
) -> Dict[str, Any]:
    example["text"] = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": notus_critique_instruction_template.format(
                    instruction=example["instruction"],
                    response=example["response"]
                ),
            },
            {
                "role": "assistant",
                "content": score_given_template.format(
                    score=example["overall_score"],
                    critique=example["critique"]
                )
            },
        ],
        tokenize=False,
        add_generation_prompt=False,  # Set to True only for generation
    )
    return example

tokenizer = AutoTokenizer.from_pretrained(model_name)

column_names = list(uf_critique.column_names)

uf_critique = uf_critique.map(
    apply_chat_template_and_prepare_for_sft,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=4,
    remove_columns=column_names,
    desc="Formatting responses with prompt template",
)

print(f"Saving to file dataset: {local_path}")
uf_critique.to_json(local_path)

data_files = {"train": local_path}

uf_critique = load_dataset("json", data_files=data_files, split="train")
print("File loaded correctly")
