import os
import sys
import argparse
import logging
from typing import Any, List
import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
import warnings

# Suppress warnings and set logging level
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

PROMPT = """Read and analyze the following text, then you need to provide your reasoning within <think></think> tags. Finally, generate a comprehensive understanding of this text."""


class VLLMThinkingWrapper:
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.forward_model_path = kwargs.get("forward_model_path", model_path)
        self.device_map = kwargs.get("device_map", "auto")
        self.torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        self.custom_prompt = kwargs.get("custom_prompt", PROMPT)
        self.use_vllm_thinking: bool = kwargs.get("use_vllm_thinking", True)
        if not self.use_vllm_thinking:
            logger.info(
                "[Config] use_vllm_thinking = False  Will skip vLLM generation and mimic RLHFDataset prompts during encoding"
            )
        else:
            logger.info(
                "[Config] use_vllm_thinking = True  Will generate extra thinking with vLLM before encoding"
            )
        self.input_max_length = kwargs.get("input_max_length", 1024)
        self.thinking_max_length = kwargs.get("thinking_max_length", 4096)
        self.vllm_tensor_parallel_size = kwargs.get(
            "vllm_tensor_parallel_size",
            int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "2")),
        )
        self.vllm_gpu_memory_utilization = kwargs.get(
            "vllm_gpu_memory_utilization",
            float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.8")),
        )
        self.vllm_max_model_len = kwargs.get(
            "vllm_max_model_len", self.input_max_length + self.thinking_max_length
        )
        self.pooling_method = kwargs.get("pooling_method", "mean").lower()
        self.batch_size = kwargs.get("batch_size", 8)
        self.truncation = kwargs.get("truncation", "right")
        print(f"VLLMThinkingWrapper Configuration:")
        print(f"  - Model: {os.path.basename(self.model_path)}")
        print(
            f"  - Tensor parallel: {self.vllm_tensor_parallel_size}, Memory: {self.vllm_gpu_memory_utilization}"
        )
        print(
            f"  - Max lengths: input={self.input_max_length}, thinking={self.thinking_max_length}"
        )
        print(f"  - Pooling: {self.pooling_method}, Batch size: {self.batch_size}")
        self._init_models()

    def _init_models(self):
        if self.use_vllm_thinking:
            self._init_vllm_model()
        else:
            self.vllm_model = None
        self._init_forward_model()
        self._init_tokenizer()

    def _init_vllm_model(self):
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoConfig
            import os

            print("Initializing vLLM model...")
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            vllm_kwargs = {
                "model": self.model_path,
                "tensor_parallel_size": self.vllm_tensor_parallel_size,
                "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
                "max_model_len": self.vllm_max_model_len,
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "enforce_eager": True,
                "disable_custom_all_reduce": True,
            }
            if hasattr(config, "model_type") and config.model_type in ["qwen2", "qwen"]:
                vllm_kwargs["max_seq_len_to_capture"] = min(
                    self.vllm_max_model_len, 2048
                )
            self.vllm_model = LLM(**vllm_kwargs)
            self.sampling_params = SamplingParams(
                temperature=1.0,
                max_tokens=self.thinking_max_length,
                top_p=1.0,
                stop_token_ids=None,
            )
            print("✓ vLLM model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM model: {e}")
            raise e

    def _init_forward_model(self):
        try:
            from transformers import AutoModelForCausalLM

            self.forward_model = AutoModelForCausalLM.from_pretrained(
                self.forward_model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            print("✓ Forward model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize forward model: {e}")
            raise e

    def _init_tokenizer(self):
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if (
                hasattr(self.tokenizer, "chat_template")
                and self.tokenizer.chat_template is not None
            ):
                test_messages = [{"content": "Hello", "role": "user"}]
                try:
                    test_prompt = self.tokenizer.apply_chat_template(
                        test_messages, add_generation_prompt=True, tokenize=False
                    )
                except Exception as e:
                    logger.warning(f"Chat template test failed: {e}")
            else:
                logger.warning(
                    "Chat template not available, will use simple concatenation"
                )
            print("✓ Tokenizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            raise e

    def _find_subsequence(self, main_list: List[Any], sub_list: List[Any]) -> int:
        len_sub = len(sub_list)
        for i in range(len(main_list) - len_sub + 1):
            if main_list[i : i + len_sub] == sub_list:
                return i
        return -1

    def _get_system_prompt_token_length(self) -> int:
        if hasattr(self, "_system_prompt_token_length_val"):
            return self._system_prompt_token_length_val
        try:
            dummy_user_content = "some_unique_string_for_testing_user_content"
            messages = [
                {"role": "system", "content": self.custom_prompt},
                {"role": "user", "content": dummy_user_content},
            ]
            full_prompt_str = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            full_tokens = self.tokenizer.encode(
                full_prompt_str, add_special_tokens=False
            )
            user_content_tokens = self.tokenizer.encode(
                dummy_user_content, add_special_tokens=False
            )
            user_content_start_index = self._find_subsequence(
                full_tokens, user_content_tokens
            )
            if user_content_start_index == -1:
                logger.warning(
                    "Could not reliably determine system prompt length for masking. "
                    "The tokenizer might be altering the user content string. "
                    "Pooling will use the full sequence (including system prompt)."
                )
                self._system_prompt_token_length_val = 0
            else:
                self._system_prompt_token_length_val = user_content_start_index
        except Exception as e:
            logger.error(
                f"Error calculating system prompt length: {e}. Defaulting to 0."
            )
            self._system_prompt_token_length_val = 0
        return self._system_prompt_token_length_val

    def _text_to_messages(self, text: str) -> List[dict]:
        return [
            {
                "role": "system",
                "content": PROMPT,
            },
            {
                "role": "user",
                "content": text,
            },
        ]

    def _apply_chat_template_safe(self, messages: List[dict]) -> str:
        try:
            if (
                hasattr(self.tokenizer, "chat_template")
                and self.tokenizer.chat_template is not None
            ):
                return self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            else:
                content = messages[0]["content"] if messages else ""
                return f"{content}\n\nAssistant:"
        except Exception as e:
            logger.warning(f"Chat template failed, using fallback: {e}")
            content = messages[0]["content"] if messages else ""
            return f"{content}\n\nAssistant:"

    def _generate_responses(self, texts: List[str], return_prompts: bool = False):
        prompts = []
        for text in texts:
            messages = self._text_to_messages(text)
            chat_prompt = self._apply_chat_template_safe(messages)
            tokens = self.tokenizer.encode(chat_prompt, add_special_tokens=False)
            if len(tokens) > self.input_max_length:
                tokens = tokens[: self.input_max_length]
                truncated_prompt = self.tokenizer.decode(
                    tokens, skip_special_tokens=True
                )
            else:
                truncated_prompt = chat_prompt
            prompts.append(truncated_prompt)
        try:
            outputs = self.vllm_model.generate(prompts, self.sampling_params)
            responses = []
            for output in outputs:
                generated_text = output.outputs[0].text
                thinking_tokens = self.tokenizer.encode(
                    generated_text, add_special_tokens=False
                )
                if len(thinking_tokens) > self.thinking_max_length:
                    thinking_tokens = thinking_tokens[: self.thinking_max_length]
                    truncated_thinking = self.tokenizer.decode(
                        thinking_tokens, skip_special_tokens=True
                    )
                else:
                    truncated_thinking = generated_text
                responses.append(truncated_thinking)
            if return_prompts:
                return responses, prompts
            return responses
        except Exception as e:
            logger.error(f"Failed to generate responses: {e}")
            empty_responses = [""] * len(texts)
            if return_prompts:
                return empty_responses, prompts
            return empty_responses

    def _get_embeddings(self, combined_texts: List[str]) -> np.ndarray:
        system_prompt_len = self._get_system_prompt_token_length()
        try:
            all_embeddings = []
            max_embedding_length = min(self.vllm_max_model_len, 6144)
            for i in range(0, len(combined_texts), self.batch_size):
                batch_texts = combined_texts[i : i + self.batch_size]
                batch_encodings = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=True,
                    truncation=False,
                )
                input_ids, attention_mask = verl_F.postprocess_data(
                    input_ids=batch_encodings["input_ids"],
                    attention_mask=batch_encodings["attention_mask"],
                    max_length=max_embedding_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation,
                )
                effective_lengths = attention_mask.sum(dim=1)
                batch_max_effective = effective_lengths.max().item()
                if batch_max_effective < input_ids.shape[1]:
                    original_length = input_ids.shape[1]
                    batch_start_pos, batch_end_pos = original_length, 0
                    for j in range(input_ids.shape[0]):
                        valid_positions = torch.where(attention_mask[j] == 1)[0]
                        if len(valid_positions) > 0:
                            batch_start_pos = min(
                                batch_start_pos, valid_positions[0].item()
                            )
                            batch_end_pos = max(
                                batch_end_pos, valid_positions[-1].item()
                            )
                    if batch_start_pos < batch_end_pos:
                        effective_range_length = batch_end_pos - batch_start_pos + 1
                        input_ids = input_ids[:, batch_start_pos : batch_end_pos + 1]
                        attention_mask = attention_mask[
                            :, batch_start_pos : batch_end_pos + 1
                        ]
                position_ids = compute_position_id_with_mask(attention_mask)
                device = next(self.forward_model.parameters()).device
                inputs = {
                    "input_ids": input_ids.to(device),
                    "attention_mask": attention_mask.to(device),
                    "position_ids": position_ids.to(device),
                }
                with torch.no_grad():
                    try:
                        outputs = self.forward_model(
                            **inputs, output_hidden_states=True
                        )
                    except TypeError as e:
                        if "position_ids" in str(e):
                            logger.warning(
                                "Model forward does not accept position_ids, retrying without it."
                            )
                            inputs.pop("position_ids", None)
                            outputs = self.forward_model(
                                **inputs, output_hidden_states=True
                            )
                        else:
                            raise
                    hidden_states = outputs.hidden_states[-1]
                    pooling_mask = inputs["attention_mask"].clone()
                    if system_prompt_len > 0 and pooling_mask.shape[0] > 0:
                        for k in range(pooling_mask.shape[0]):
                            valid_token_indices = torch.where(pooling_mask[k] == 1)[0]
                            if len(valid_token_indices) > 0:
                                valid_start_pos = valid_token_indices[0].item()
                                valid_end_pos = valid_token_indices[-1].item()
                                system_prompt_end_pos = (
                                    valid_start_pos + system_prompt_len
                                )
                                if system_prompt_end_pos <= valid_end_pos + 1:
                                    pooling_mask[
                                        k, valid_start_pos:system_prompt_end_pos
                                    ] = 0
                                    logger.debug(
                                        f"Batch {i//self.batch_size + 1}, item {k}: excluded system prompt tokens {valid_start_pos}:{system_prompt_end_pos} from valid range {valid_start_pos}:{valid_end_pos+1}"
                                    )
                                else:
                                    logger.warning(
                                        f"Batch {i//self.batch_size + 1}, item {k}: system prompt length ({system_prompt_len}) exceeds valid token length ({len(valid_token_indices)}). Skipping exclusion."
                                    )
                            else:
                                logger.warning(
                                    f"Batch {i//self.batch_size + 1}, item {k}: no valid tokens found in attention mask."
                                )
                    if self.pooling_method == "mean":
                        masked_hidden_states = hidden_states * pooling_mask.unsqueeze(
                            -1
                        )
                        sum_hidden_states = masked_hidden_states.sum(dim=1)
                        token_counts = pooling_mask.sum(dim=1, keepdim=True)
                        token_counts = torch.clamp(token_counts, min=1)
                        pooled = sum_hidden_states / token_counts
                    elif self.pooling_method == "eos":
                        lengths = pooling_mask.sum(dim=1) - 1
                        lengths = torch.clamp(lengths, min=0)
                        batch_indices = torch.arange(
                            hidden_states.size(0), device=hidden_states.device
                        )
                        pooled = hidden_states[batch_indices, lengths, :]
                    else:
                        raise ValueError(
                            f"Unsupported pooling_method: {self.pooling_method}"
                        )
                    embeddings = pooled.cpu().float().numpy()
                    all_embeddings.append(embeddings)
            final_embeddings = np.vstack(all_embeddings)
            return final_embeddings
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            import traceback

            traceback.print_exc()
            raise e

    def _ensure_models_initialized(self):
        if self.use_vllm_thinking and (
            not hasattr(self, "vllm_model") or self.vllm_model is None
        ):
            logger.warning(
                "vLLM model not found or lost (likely after serialization). Reinitializing vLLM model..."
            )
            self._init_vllm_model()
        elif not hasattr(self, "forward_model") or self.forward_model is None:
            logger.warning(
                "Forward model not found or lost (likely after serialization). Reinitializing models..."
            )
            self._init_models()
        elif not hasattr(self, "tokenizer") or self.tokenizer is None:
            logger.warning(
                "Tokenizer not found or lost (likely after serialization). Reinitializing models..."
            )
            self._init_models()

    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        self._ensure_models_initialized()
        if self.use_vllm_thinking:
            responses, prompts = self._generate_responses(
                sentences, return_prompts=True
            )
            combined_texts = [prompt + resp for prompt, resp in zip(prompts, responses)]
        else:
            combined_texts = []
            for orig in sentences:
                messages = self._text_to_messages(orig)
                chat_prompt = self._apply_chat_template_safe(messages)
                tokens = self.tokenizer.encode(chat_prompt, add_special_tokens=False)
                if len(tokens) > self.input_max_length:
                    tokens = tokens[: self.input_max_length]
                    chat_prompt = self.tokenizer.decode(
                        tokens, skip_special_tokens=True
                    )
                combined_texts.append(chat_prompt)
        embeddings = self._get_embeddings(combined_texts)
        return embeddings

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus, **kwargs) -> np.ndarray:
        if isinstance(corpus, list):
            if len(corpus) > 0 and isinstance(corpus[0], dict):
                sentences = []
                for doc in corpus:
                    if "text" in doc:
                        sentences.append(doc["text"])
                    elif "title" in doc and "text" in doc:
                        sentences.append(f"{doc['title']} {doc['text']}")
                    else:
                        text_parts = [
                            str(v) for v in doc.values() if isinstance(v, str)
                        ]
                        sentences.append(" ".join(text_parts))
            else:
                sentences = [str(s) for s in corpus]
        elif isinstance(corpus, dict):
            sentences = []
            for key, values in corpus.items():
                if isinstance(values, list):
                    sentences.extend([str(v) for v in values])
                else:
                    sentences.append(str(values))
        else:
            sentences = [str(corpus)]
        return self.encode(sentences, **kwargs)


try:
    from mteb.encoder_interface import Encoder
    from mteb.model_meta import ModelMeta
except ImportError:
    print("MTEB not available. Please install with: pip install mteb")
    sys.exit(1)


class MTEBVLLMAdapter(Encoder):
    def __init__(self, model_path: str, model_name: str | None = None, **kwargs):
        if model_name is None:
            model_name = (
                os.path.basename(model_path.rstrip("/")) or "vllm_thinking_model"
            )
        self.model_name = model_name
        self.model_card_name = f"GRACE/{self.model_name}"
        self.revision = None
        self.model_revision = None
        self.mteb_model_meta = ModelMeta(
            name=f"GRACE/{self.model_name}",
            revision=None,
            release_date="2025-08-24",
            languages=["eng-Latn"],
            n_parameters=7000000000,
            memory_usage_mb=14000.0,
            max_tokens=4096.0,
            embed_dim=4096,
            license="mit",
            framework=["PyTorch"],
            open_weights=True,
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets={"custom": ["custom_dataset"]},
            public_training_code=None,
            public_training_data=None,
            reference=None,
        )
        self.wrapper = VLLMThinkingWrapper(model_path, **kwargs)

    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        return self.wrapper.encode(sentences, **kwargs)

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        return self.wrapper.encode_queries(queries, **kwargs)

    def encode_corpus(self, corpus, **kwargs) -> np.ndarray:
        return self.wrapper.encode_corpus(corpus, **kwargs)

    def __getattr__(self, name):
        if name in ["get_model_name", "get_name", "model_name_or_path"]:
            return self.model_name
        if name in ["get_revision", "get_model_revision"]:
            return self.revision
        if hasattr(self.wrapper, name):
            return getattr(self.wrapper, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


def create_mteb_model(model_path: str, **kwargs) -> MTEBVLLMAdapter:
    model_name = kwargs.pop("model_name", None)
    return MTEBVLLMAdapter(model_path, model_name=model_name, **kwargs)


def create_model_meta(model_name: str, model_path: str, **kwargs) -> ModelMeta:
    def loader(**loader_kwargs):
        merged_kwargs = kwargs.copy()
        merged_kwargs.update(loader_kwargs)
        return create_mteb_model(model_path, **merged_kwargs)

    return ModelMeta(
        loader=loader,
        name=model_name,
        languages=["eng-Latn"],
        open_weights=True,
        revision=None,
        release_date="2025-01-01",
        license="mit",
        framework=["PyTorch"],
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=None,
        embed_dim=None,
        similarity_fn_name=None,
        use_instructions=None,
        training_datasets=None,
        public_training_code=None,
        public_training_data=None,
        reference=None,
    )


def run_mteb_evaluation(
    model_path: str,
    task_names: List[str],
    output_dir: str = "results",
    batch_size: int = 8,
    **kwargs,
):
    try:
        import mteb
        from mteb import MTEB
    except ImportError:
        logger.error("MTEB not installed. Please install with: pip install mteb")
        return

    model_name = kwargs.get("model_name", None)
    if model_name is None:
        model_name = os.path.basename(model_path.rstrip("/")) or "vllm_thinking_model"
    kwargs_without_model_name = kwargs.copy()
    kwargs_without_model_name.pop("model_name", None)
    model = MTEBVLLMAdapter(
        model_path,
        model_name=model_name,
        batch_size=batch_size,
        **kwargs_without_model_name,
    )
    evaluation = MTEB(tasks=task_names)
    print("Running evaluation...")
    results = evaluation.run(
        model, output_folder=output_dir, eval_splits=["test"], verbosity=1
    )
    print("Evaluation completed!")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run MTEB evaluation with VLLMThinkingWrapper"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model for vLLM thinking generation",
    )
    parser.add_argument(
        "--forward_model_path",
        type=str,
        default=None,
        help="Path to the model for forward pass (defaults to model_path)",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=["STS16"], help="MTEB tasks to evaluate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for processing"
    )
    parser.add_argument(
        "--input_max_length",
        type=int,
        default=1024,
        help="Max length for prompt + input",
    )
    parser.add_argument(
        "--thinking_max_length",
        type=int,
        default=4096,
        help="Max length for thinking generation",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=None,
        help="vLLM tensor parallel size",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=None,
        help="vLLM GPU memory utilization",
    )
    parser.add_argument(
        "--pooling_method",
        type=str,
        choices=["mean", "eos"],
        default="mean",
        help="Pooling method: mean or eos",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name for result directory (defaults to basename of model_path)",
    )
    args = parser.parse_args()
    kwargs = {}
    kwargs["input_max_length"] = args.input_max_length
    kwargs["thinking_max_length"] = args.thinking_max_length
    if args.vllm_tensor_parallel_size is not None:
        kwargs["vllm_tensor_parallel_size"] = args.vllm_tensor_parallel_size
    if args.vllm_gpu_memory_utilization is not None:
        kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
    kwargs["pooling_method"] = args.pooling_method
    if args.model_name is not None:
        kwargs["model_name"] = args.model_name
    if args.forward_model_path is not None:
        kwargs["forward_model_path"] = args.forward_model_path
    results = run_mteb_evaluation(
        model_path=args.model_path,
        task_names=args.tasks,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        **kwargs,
    )
    if results:
        logger.info("Evaluation results:")
        if isinstance(results, dict):
            for task, result in results.items():
                logger.info(f"  {task}: {result}")
        elif isinstance(results, list):
            logger.info(f"  Results: {results}")
        else:
            logger.info(f"  Results: {results}")
    else:
        logger.info("No results returned")


if __name__ == "__main__":
    main()
