import argparse
import json
import os
import pandas as pd
from typing import List, Dict, Any

from verl.utils.hdfs_io import copy, makedirs


def load_data(file_path: str, mode: str) -> List[Dict[str, str]]:
    samples = []

    file_ext = os.path.splitext(file_path)[1].lower()

    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            if file_ext == ".txt" and mode == "unsupervised":
                sample = {"text": line, "source": "wiki1m", "line_idx": line_idx}
                samples.append(sample)
                continue

            try:
                sample = json.loads(line)

                if mode == "supervised":
                    required_fields = ["query", "positive", "negative"]
                    if all(key in sample for key in required_fields):
                        sample["line_idx"] = line_idx
                        if "source" not in sample:
                            sample["source"] = None
                        samples.append(sample)
                    else:
                        missing_fields = [
                            field for field in required_fields if field not in sample
                        ]
                        print(
                            f"Warning: Line {line_idx} missing required fields: {missing_fields}"
                        )

                elif mode == "unsupervised":
                    if "text" in sample:
                        sample["line_idx"] = line_idx
                        if "source" not in sample:
                            sample["source"] = None
                        samples.append(sample)
                    else:
                        print(f"Warning: Line {line_idx} missing required field: text")

            except json.JSONDecodeError as e:

                if file_ext == ".txt" and mode != "unsupervised":
                    print(
                        f"Warning: txt file format only supported for unsupervised mode"
                    )
                else:
                    print(f"Warning: Failed to parse line {line_idx}: {e}")
                continue

    print(f"Loaded {len(samples)} valid samples from {file_path}")
    return samples


PROMPT = """Read and analyze the following text, then you need to provide your reasoning within <think></think> tags. Finally, generate a comprehensive understanding of this text."""

PROMPT_ALTERNATIVE1 = """Analyze the following input and provide your reasoning within <think></think> tags. I will use the final eos token of your response after the closing </think> tag to represent the entire input.\n\nINPUT TO ANALYZE: """

PROMPT_ALTERNATIVE2 = """You are tasked with generating semantic representations for contrastive learning.

**Instructions**: First, think step-by-step about the text analysis, then provide your final representation.

**Text**: *[INPUT_TEXT]*

**Step 1: Think**
Analyze this text carefully. Consider:
- What are the core semantic concepts and main ideas?
- What is the intent, tone, and context of this text?
- What distinctive features make this text unique?
- How should these elements be weighted in the representation?

**Step 2: Generate Representation**
Based on your analysis, create an optimized semantic representation that captures the essential meaning and distinctive characteristics.

Please follow this exact format:
**Analysis**: [Your step-by-step thinking process]
**Representation**: [Your final semantic representation]"""


def process_sample_supervised(
    sample: Dict[str, str], split: str, idx: int
) -> Dict[str, Any]:

    query = sample["query"].strip()
    positive = sample["positive"].strip()
    negative = sample["negative"].strip()
    source = sample.get("source", None)

    ground_truth = positive

    processed_sample = {
        "data_source": "supervised",
        "prompt": [
            {
                "role": "system",
                "content": PROMPT,
            },
            {
                "role": "user",
                "content": positive,
            },
        ],
        "ability": "natural_language_inference",
        "reward_model": {"style": "multi_input", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": idx,
            "original_query": query,
            "positive_example": positive,
            "negative_example": negative,
            "source": source,
            "line_idx": sample.get("line_idx", idx),
        },
        "multi_input": {
            "query": query,
            "instruction": positive,
            "negative_document": negative,
        },
    }

    return processed_sample


def process_sample_unsupervised(
    sample: Dict[str, str], split: str, idx: int
) -> Dict[str, Any]:
    text = sample["text"].strip()
    source = sample.get("source", None)

    processed_sample = {
        "data_source": "unsupervised",
        "prompt": [
            {
                "role": "system",
                "content": PROMPT,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        "ability": "natural_language_inference",
        "reward_model": {"style": "unsupervised", "ground_truth": ""},
        "extra_info": {
            "split": split,
            "index": idx,
            "text": text,
            "source": source,
            "line_idx": sample.get("line_idx", idx),
        },
    }

    return processed_sample


def split_dataset(samples: List[Dict[str, str]], test_ratio: float = 0.1) -> tuple:
    total_samples = len(samples)
    test_size = int(total_samples * test_ratio)

    test_samples = samples[:test_size]
    train_samples = samples[test_size:]

    print(f"Dataset split: Train={len(train_samples)}, Test={len(test_samples)}")

    return train_samples, test_samples


def process_and_save_split(
    samples: List[Dict[str, str]], split_name: str, output_dir: str, mode: str
):
    processed_samples = []

    for idx, sample in enumerate(samples):
        try:
            if mode == "supervised":
                processed_sample = process_sample_supervised(sample, split_name, idx)
            else:  # unsupervised
                processed_sample = process_sample_unsupervised(sample, split_name, idx)
            processed_samples.append(processed_sample)
        except Exception as e:
            print(f"Warning: Failed to process sample {idx} in {split_name}: {e}")
            continue

    df = pd.DataFrame(processed_samples)
    parquet_path = os.path.join(output_dir, f"{split_name}.parquet")
    df.to_parquet(parquet_path, index=False)

    json_path = os.path.join(output_dir, f"{split_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(processed_samples, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(processed_samples)} {split_name} samples to {parquet_path}")
    print(f"Saved {len(processed_samples)} {split_name} samples to {json_path}")
    return len(processed_samples)


def main():
    parser = argparse.ArgumentParser(description="data processing script")
    parser.add_argument(
        "--mode",
        choices=["supervised", "unsupervised"],
        required=True,
        help="supervised (need query/positive/negative) or unsupervised (need text)",
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="input file path (jsonl for both modes, txt for unsupervised mode only)",
    )
    parser.add_argument("--local_dir", required=True, help="local output directory")
    parser.add_argument(
        "--hdfs_dir", default=None, help="HDFS output directory (optional)"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.01, help="test data ratio"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="maximum number of samples (for testing)",
    )

    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    print(f"Loading dataset from {args.input_file} in {args.mode} mode...")
    samples = load_data(args.input_file, args.mode)

    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        print(f"Limited to {len(samples)} samples for testing")

    train_samples, test_samples = split_dataset(samples, args.test_ratio)

    total_processed = 0

    if train_samples:
        total_processed += process_and_save_split(
            train_samples, "train", args.local_dir, args.mode
        )

    if test_samples:
        total_processed += process_and_save_split(
            test_samples, "test", args.local_dir, args.mode
        )

    print(f"\nProcessing complete! Total processed samples: {total_processed}")

    if args.hdfs_dir is not None:
        print(f"Copying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
        print("HDFS copy complete!")

    print("\n" + "=" * 50)
    print("Sample processed data:")
    print("=" * 50)

    if train_samples:
        if args.mode == "supervised":
            sample_processed = process_sample_supervised(train_samples[0], "train", 0)
            print(f"Mode: {args.mode}")
            print(f"Original query: {train_samples[0]['query']}")
            print(f"Original positive: {train_samples[0]['positive']}")
            print(f"Original negative: {train_samples[0]['negative']}")
            print(f"Original source: {train_samples[0].get('source', 'None')}")
            print(
                f"\nProcessed instruction: {sample_processed['prompt'][0]['content']}"
            )
            print(f"Ground truth: {sample_processed['reward_model']['ground_truth']}")
            print(f"Multi-input query: {sample_processed['multi_input']['query']}")
            print(
                f"Multi-input negative: {sample_processed['multi_input']['negative_document']}"
            )
            print(
                f"Multi-input positive: {sample_processed['multi_input']['instruction']}"
            )
            print(f"Source info: {sample_processed['extra_info']['source']}")
        else:  # unsupervised
            sample_processed = process_sample_unsupervised(train_samples[0], "train", 0)
            print(f"Mode: {args.mode}")
            print(f"Original text: {train_samples[0]['text']}")
            print(f"Original source: {train_samples[0].get('source', 'None')}")
            print(f"\nProcessed sample:")
            print(json.dumps(sample_processed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
