import argparse
import time
import datasets
import typing as t
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from transformers.utils import check_min_version as check_min_transformers_version
from optimum.habana.utils import check_optimum_habana_min_version

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def setup_env() -> None:
    check_min_transformers_version("4.34.0")
    check_optimum_habana_min_version("1.9.0.dev0")

    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

    adapt_transformers_to_gaudi()

def load_model_and_tokenizer(args: argparse.ArgumentParser) -> t.Tuple[AutoModelForCausalLM, t.Any]:
    """Loads the model and tokenizer used to run the inference."""
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float

    logger.info(f"Using {torch_dtype} type")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch_dtype
    ).eval().to(args.device)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    if args.use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        logger.info("Using HPU graphs")

        model = wrap_in_hpu_graph(model)

    return model, tokenizer

def load_dataset_for_inference(args: argparse.ArgumentParser) -> t.Any:
    return datasets.load_dataset(args.dataset_name, args.dataset_subset, split=args.dataset_split)


def setup_parser(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to the pre-trained model. It can be a Hugging Face Hub repo id or a path to a local directory."
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        choices=["hpu", "cpu"],
        help="Torch device to run the computations", 
        default="hpu"
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies",
        default=True,
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation using bfloat16 precision",
        default=True,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="The name of the dataset to run inference for",
        required=True,
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        help="The name of the subset to load from the dataset",
        default=None,
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        help="The name of the split to load from the dataset",
        default=None,
    )
    parser.add_argument(
        "--dataset_column",
        type=str,
        help="The name of the column of the dataset containing the instructions"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="The number of max new tokens to generate.",
        default=128,
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to do generation sampling or not.",
        default=True,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        help="The percentage of next token top probabilities to consider when sampling",
        default=1.0
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        help="The prompt format to be used. It should contain a placeholder called 'instruction'",
        default="<|system|>\n</s>\n<|user|>\n{instruction}</s>\n<|assistant|>"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results.jsonl",
        help="The file where the results will be saved in JSONL format"
    )

    return parser.parse_args()

def main() -> None:
    parser = argparse.ArgumentParser()
    args = setup_parser(parser)

    logger.info(f"Running inference with: {args}")

    setup_env()

    model, tokenizer = load_model_and_tokenizer(args)
    dataset = load_dataset_for_inference(args)

    def generate(example: dict) -> str:
        instruction = example[args.dataset_column]
        prompt = args.prompt_format.format(instruction=instruction) 

        input_tokens = tokenizer.batch_encode_plus(
            [prompt],
            return_tensors="pt",
            padding=True
        ).to(args.device)

        with torch.no_grad():
            generation_output = model.generate(
                **input_tokens,
                do_sample=args.do_sample,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=args.max_new_tokens,
                lazy_mode=True,
                hpu_graphs=args.use_hpu_graphs,
                ignore_eos=False,
                top_p=args.top_p,
                temperature=args.temperature,
            )

        generation = tokenizer.batch_decode(
            generation_output.sequences[:, input_tokens.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        example["output"] = generation
        example["generator"] = args.model_name_or_path

        return example

    start = time.perf_counter()
    dataset = dataset.map(generate)
    end = time.perf_counter()

    elapsed = end - start
    logger.info(f"Generation finished. Took: {elapsed:.3f}s")

    dataset.to_json(args.output_file)
    logger.info(f"Wrote results to '{args.output_file}'!")

if __name__ == "__main__":
    main()
