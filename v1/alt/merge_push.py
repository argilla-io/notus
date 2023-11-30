# Script adapted from https://github.com/jondurbin/qlora/blob/main/merge.py
# This script is used to merge the base model and the PEFT model and upload the result to the Hub,
# with the option to also upload the model if not PEFT adapter.
# Usage: python merge_push.py --base <base_model> --peft <peft_model> --out <output_model> --push --merge --include-eval

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str)
    parser.add_argument("--peft", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--include-eval", action="store_true")

    return parser.parse_args()


def main():
    args = get_args()
    print(f"Args: {args}")
    print(f"Loading base model: {args.base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Loading PEFT: {args.peft}")
    model = PeftModel.from_pretrained(base_model, args.peft)

    if args.merge:
        print("Running merge_and_unload")
        model = model.merge_and_unload()

    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.save_pretrained(args.out)

    if args.push:
        print("Saving to hub ...")
        model.push_to_hub(args.out, private=True)
        tokenizer.push_to_hub(args.out, private=True)

    if args.include_eval:
        from huggingface_hub import HfApi

        api = HfApi()
        for file in ["all_results.json", "eval_results.json"]:
            try:
                api.upload_file(
                    path_or_fileobj=f"{args.peft}/{file}",
                    path_in_repo=file,
                    repo_id=args.out,
                    repo_type="model",
                )
            except Exception as e:
                print(f"Failed to upload {file}: {e}")


if __name__ == "__main__":
    main()
