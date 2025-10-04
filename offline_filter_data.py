#!/usr/bin/env python3
import argparse
import os
from typing import Dict, Optional

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def _filter_dataset(
    parquet_path: str,
    out_path: str,
    tokenizer,
    max_len: int,
    prompt_key: str = "prompt",
    num_proc: int = 4,
):
    print(f"[INFO] Loading {parquet_path} …")
    ds: Dataset = load_dataset("parquet", data_files=parquet_path)["train"]
    before_n = len(ds)

    def _len_ok(example):
        return (
            len(
                tokenizer.apply_chat_template(
                    example[prompt_key], add_generation_prompt=True
                )
            )
            <= max_len
        )

    print(f"[INFO] Filtering samples with length > {max_len} …")
    ds = ds.filter(_len_ok, num_proc=num_proc, desc="len_filter")
    after_n = len(ds)
    kept_ratio = after_n / before_n * 100.0
    print(f"[DONE] {after_n}/{before_n} samples kept ({kept_ratio:.2f}%).")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[INFO] Writing filtered dataset to {out_path} …")
    ds.to_pandas().to_parquet(out_path, index=False)
    print("[OK] Saved.")


def main():
    parser = argparse.ArgumentParser(
        description="Offline token-length filtering for Echo parquet data"
    )
    parser.add_argument("--train_parquet", required=True, help="Path to train.parquet")
    parser.add_argument(
        "--val_parquet", default=None, help="Path to val/test parquet (optional)"
    )
    parser.add_argument(
        "--out_dir", required=True, help="Directory to write filtered parquet files"
    )
    parser.add_argument(
        "--tokenizer_path",
        required=True,
        help="HF checkpoint used in training (must support chat template)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1024,
        help="Maximum allowed prompt length (tokens)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=os.cpu_count() // 2 or 1,
        help="Workers for datasets.filter",
    )
    parser.add_argument(
        "--prompt_key",
        default="prompt",
        help="Column name of prompt messages in the parquet",
    )
    args = parser.parse_args()

    print("[INFO] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )

    out_train = os.path.join(args.out_dir, "train.parquet")
    _filter_dataset(
        args.train_parquet,
        out_train,
        tokenizer,
        args.max_len,
        args.prompt_key,
        args.num_proc,
    )

    if args.val_parquet:
        out_val = os.path.join(args.out_dir, "test.parquet")
        _filter_dataset(
            args.val_parquet,
            out_val,
            tokenizer,
            args.max_len,
            args.prompt_key,
            args.num_proc,
        )


if __name__ == "__main__":
    main()
