# Assume necessary imports are present
import os
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Generator, Callable # Added Generator, Callable

import torch
import torch.nn.functional as F

# Ensure these classes/modules exist in your project structure

from model.args import ModelArgs
from model.model import LLM # Assuming Transformer is the actual model class used
from model.datatypes import TransformerInput, LLMInput, GenerationResult, RawContent, RawMessage, MaskedEmbedding # Added MaskedEmbedding
from model.tokenizer import Tokenizer # Assuming Tokenizer class exists and has get_instance()
from chat_format import ChatFormat # Assuming this handles chat formatting

    
# Placeholder for QuantizationMode if used
class QuantizationMode:
    fp8_mixed = "fp8_mixed"
    int4_mixed = "int4_mixed"
    # Add other modes if necessary

# --- TinyLlama Class ---
class TinyLlama:
    """
    Wrapper class for the TinyLlama model providing generation capabilities.
    Handles model loading, tokenization, and text generation loops.
    """
    def __init__(self, model: LLM, tokenizer: Tokenizer, args: ModelArgs):
        """
        Initializes the TinyLlama wrapper.

        Args:
            model: The loaded Transformer model instance.
            tokenizer: The tokenizer instance.
            args: Model arguments/configuration.
        """
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        # Initialize ChatFormat, handling potential absence of vision_args
        self.formatter = ChatFormat(tokenizer, vision_args=getattr(args, 'vision_args', None))

    @staticmethod
    def build(
        ckpt_dir: str,
        max_seq_len: int,
        max_batch_size: int,
        quantization_mode: Optional[str] = None, # Use string representation of QuantizationMode
        seed: int = 1,
    ) -> 'TinyLlama': # Return type is the class itself
        """
        Builds and loads the TinyLlama model from a checkpoint directory.

        Args:
            ckpt_dir: Directory containing checkpoint files (*.pth) and params.json.
            max_seq_len: Maximum sequence length for the model.
            max_batch_size: Maximum batch size for inference.
            quantization_mode: Optional quantization mode (e.g., "fp8_mixed", "int4_mixed").
            seed: Random seed for reproducibility.

        Returns:
            An instance of the TinyLlama class.

        Raises:
            FileNotFoundError: If checkpoint or params.json is not found.
            AssertionError: If vocabulary sizes mismatch or no checkpoints are found.
            ImportError: If quantization is requested but loader cannot be imported.
        """
        # Set device explicitly (e.g., cuda if available, otherwise cpu)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            # If multiple GPUs are available, default to the first one (index 0).
            # User might need to set CUDA_VISIBLE_DEVICES environment variable
            # to select a specific GPU if not using distributed setup.
            torch.cuda.set_device(0)
        print(f"Using device: {device}")

        torch.manual_seed(seed)

        start_time = time.time()

        ckpt_paths = sorted(Path(ckpt_dir).glob("*.pth"))
        if not ckpt_paths: # More explicit check
             raise FileNotFoundError(f"No checkpoint files (*.pth) found in {ckpt_dir}")

        # --- Logic adjusted for single checkpoint loading ---
        # Assuming non-sharded setup. If multiple .pth files exist,
        # load the first one alphabetically.
        if len(ckpt_paths) > 1:
            print(f"Warning: Found multiple checkpoint files ({len(ckpt_paths)}) in {ckpt_dir}. Loading the first one: {ckpt_paths[0]}")
            print("         Consider providing a directory with a single consolidated checkpoint for clarity.")
        ckpt_path = ckpt_paths[0]
        print(f"Loading checkpoint from: {ckpt_path}")

        params_path = Path(ckpt_dir) / "params.json"
        if not params_path.is_file():
             raise FileNotFoundError(f"params.json not found in {ckpt_dir}")
        with open(params_path, "r") as f:
            params = json.loads(f.read())

        # Instantiate ModelArgs safely
        try:
            model_args: ModelArgs = ModelArgs(
                **params,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
            )
        except TypeError as e:
            print(f"Error initializing ModelArgs: {e}")
            print("Please ensure 'params.json' contains the expected arguments for ModelArgs.")
            raise

        # Assuming Tokenizer is a singleton or has a static getter
        # Ensure Tokenizer is properly initialized before this point if needed.
        try:
             tokenizer = Tokenizer.get_instance()
        except Exception as e:
             print(f"Error getting Tokenizer instance: {e}")
             print("Ensure Tokenizer class is defined and get_instance() works.")
             raise

        # Validate vocab size
        if model_args.vocab_size == -1:
            model_args.vocab_size = tokenizer.n_words
        assert model_args.vocab_size == tokenizer.n_words, \
            f"Vocabulary size mismatch: model args ({model_args.vocab_size}) vs tokenizer ({tokenizer.n_words})"
        # Use model_dump_json if available (pydantic v2), else fallback
        if hasattr(model_args, 'model_dump_json'):
             print("Model args:\n", model_args.model_dump_json(indent=2))
        elif hasattr(model_args, 'json'):
             print("Model args:\n", model_args.json(indent=2))
        else:
             print("Model args:\n", vars(model_args))


        # --- State dict loading adjusted for single checkpoint ---
        print(f"Loading state dict from {ckpt_path}...")
        # Load to CPU first to avoid potential GPU OOM for large models/metadata
        state_dict = torch.load(ckpt_path, map_location="cpu")
        print("Loaded checkpoint state dict to CPU.")


        # Determine default tensor type based on device and quantization
        model = None # Initialize model variable
        if quantization_mode is not None:
            # Quantization might require specific handling or libraries
            try:
                # Note: Relative import depends on project structure.
                from .quantization.loader import convert_to_quantized_model
                print("Quantization requested.")
                # Quantized models might prefer specific types initially
                torch.set_default_dtype(torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32)
                model = Transformer(model_args) # Initialize model with default dtype
                print("Loading state dict into base model (before quantization)...")
                # Load state dict before quantization
                # Use strict=False as quantization might change model structure slightly
                model.load_state_dict(state_dict, strict=False)
                print("Done loading state dict.")
                print("Converting to quantized model...")
                model = convert_to_quantized_model(model, ckpt_dir, quantization_mode)
                # Quantized model might need specific device placement after conversion
                model.to(device)
                print(f"Quantized model moved to {device}.")

            except ImportError:
                print(f"Error: Quantization mode '{quantization_mode}' requested, but failed to import 'convert_to_quantized_model'.")
                print("       Ensure the quantization library is installed and the import path is correct.")
                raise
            except Exception as e:
                 print(f"Error during quantization process: {e}")
                 raise

        else:
             # Standard (non-quantized) loading
            print("No quantization requested.")
             # Set default tensor type based on CUDA capability for model initialization
            if device == "cuda":
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                    print("Using BFloat16 tensors.")
                else:
                    dtype = torch.float16
                    print("Using Float16 (Half) tensors.")
                torch.set_default_dtype(dtype) # Set default for new tensors
                torch.set_default_tensor_type(torch.cuda.BFloat16Tensor if dtype == torch.bfloat16 else torch.cuda.HalfTensor)
            else:
                 dtype = torch.float32
                 print("Using Float32 tensors on CPU.")
                 torch.set_default_dtype(dtype)
                 torch.set_default_tensor_type(torch.FloatTensor)

            # Initialize model with the chosen dtype
            model = Transformer(model_args)
            # Move model to the target device *before* loading state dict
            model.to(device)
            print(f"Model initialized on {device}.")
            print(f"Loading state dict onto {device}...")
            # Load state_dict directly onto the model's device
            # `assign=True` can improve performance by avoiding extra copies (requires PyTorch >= 1.9)
            try:
                 model.load_state_dict(state_dict, strict=False, assign=True)
            except TypeError: # Fallback for older PyTorch versions
                 print("Warning: assign=True not supported, loading state dict with potential extra copy.")
                 model.load_state_dict(state_dict, strict=False)
            print("Done loading state dict.")

        # Clean up CPU copy of state_dict
        del state_dict
        if device == "cuda":
            torch.cuda.empty_cache() # Free up cached memory on GPU

        print(f"Model loaded to {device} in {time.time() - start_time:.2f} seconds")

        # Instantiate and return the TinyLlama wrapper class
        return TinyLlama(model, tokenizer, model_args)


    @torch.inference_mode()
    def generate(
        self,
        llm_inputs: List[LLMInput],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
        print_model_input: bool = False,
        logits_processor: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Generator[List[GenerationResult], None, None]:
        """
        Generates text based on the provided inputs using the loaded model.

        Args:
            llm_inputs: A list of inputs, each containing tokens and optional images.
            temperature: Sampling temperature. 0 means greedy decoding (argmax). Higher values increase randomness.
            top_p: Nucleus sampling probability. Filters vocabulary to the smallest set whose cumulative probability exceeds top_p. 0.9 is common.
            max_gen_len: Maximum number of new tokens to generate per input. Defaults to model's max sequence length minus 1.
            logprobs: Whether to calculate and return log probabilities of the generated tokens.
            echo: If True, yields the input tokens as part of the generation stream before generating new tokens.
            print_model_input: If True, prints the decoded input tokens to the console for debugging.
            logits_processor: An optional callable function to modify the logits distribution before sampling.
                              Expected signature: `func(input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor`

        Yields:
            A list of GenerationResult objects, one for each input in the batch, at each generation step.

        Returns:
             A generator yielding lists of GenerationResult objects.
        """
        # Determine the effective maximum generation length
        if max_gen_len is None or max_gen_len <= 0:
             max_gen_len = self.model.args.max_seq_len - 1
        else:
             # Ensure max_gen_len doesn't exceed model limits when added to prompt
             max_gen_len = min(max_gen_len, self.model.args.max_seq_len - 1)


        params = self.model.args
        device = next(self.model.parameters()).device # Get device model is on

        # Simple print replacement for cprint (used if print_model_input is True)
        def cprint(text, color):
            # Basic color printing (might not work on all terminals)
            color_codes = {"yellow": "\033[93m", "grey": "\033[90m", "red": "\033[91m", "end": "\033[0m"}
            start_code = color_codes.get(color, "")
            end_code = color_codes.get("end", "") if start_code else ""
            print(f"{start_code}{text}{end_code}")

        # Handle printing input tokens if requested
        print_model_input = print_model_input or os.environ.get("LLAMA_MODELS_DEBUG", "0") == "1"
        if print_model_input:
            cprint("Input to model:\n", "yellow")
            for i, inp in enumerate(llm_inputs):
                try:
                    # Ensure tokens are integers or longs before decoding
                    tokens_to_decode = [int(t) for t in inp.tokens]
                    cprint(f" Batch item {i}: {self.tokenizer.decode(tokens_to_decode)}", "grey")
                except Exception as e:
                    cprint(f" Batch item {i}: Error decoding input tokens - {e}", "red")
        prompt_tokens = [inp.tokens for inp in llm_inputs]

        bsz = len(llm_inputs)
        assert bsz <= params.max_batch_size, f"Batch size {bsz} exceeds model's max_batch_size {params.max_batch_size}"

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)

        # Check if prompt itself is too long
        if max_prompt_len >= params.max_seq_len:
            cprint(f"Error: Maximum prompt length ({max_prompt_len}) exceeds model's maximum sequence length ({params.max_seq_len}). Cannot generate.", "red")
            # Optionally raise an error or return an empty generator
            return # Stop generation

        # Calculate total length: prompt + generated tokens, capped by model max length
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        # Prepare token tensor (filled with padding)
        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        # Fill with prompt tokens
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        # Initialize tensor for log probabilities if requested
        token_logprobs = None
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float, device=device)

        # Keep track of which sequences have reached EOS
        eos_reached = torch.tensor([False] * bsz, device=device)
        # Mask indicating input tokens (True) vs generated tokens (False)
        input_text_mask = tokens != pad_id
        # Pre-calculate stop tokens tensor
        stop_tokens = torch.tensor(self.tokenizer.stop_tokens, device=device)


        # --- Echo phase (yield input tokens) ---
        if echo:
            for i in range(max_prompt_len):
                # Only yield if the token is not padding for that batch item
                current_tokens = tokens[:, i]
                valid_mask = input_text_mask[:, i]
                results = []
                for j in range(bsz):
                     if valid_mask[j]:
                         # Logprobs for input tokens are usually not calculated/meaningful, set to None
                         current_logprobs_val = None
                         # If logprobs were pre-calculated for the prompt (e.g., scoring), they could be added here.
                         # For standard generation 'echo', input logprobs are typically not provided.
                         results.append(
                            GenerationResult(
                                token=current_tokens[j].item(),
                                text=self.tokenizer.decode([current_tokens[j].item()]),
                                source="input",
                                logprobs=current_logprobs_val, # Logprobs for input usually None
                                batch_idx=j,
                                finished=False, # Not finished during echo phase
                                ignore_token=False, # Don't ignore echoed input tokens
                            )
                        )
                     else:
                         # Handle cases where shorter prompts exist in the batch during echo
                         results.append(
                             GenerationResult(
                                 token=pad_id, # Or another indicator
                                 text="",
                                 source="input_padding",
                                 logprobs=None,
                                 batch_idx=j,
                                 finished=False,
                                 ignore_token=True, # Ignore padding
                             )
                         )

                if any(valid_mask): # Only yield if there was at least one valid token at this position
                    yield results

        # --- Generation phase ---
        prev_pos = 0 # Position marker for KV cache
        # Start generating from the end of the shortest prompt
        for cur_pos in range(min_prompt_len, total_len):
            # --- Handle Vision Input (if applicable) ---
            # This part assumes vision embeddings are processed only once at the beginning
            image_embedding = None
            if prev_pos == 0: # Only process images on the first forward pass
                # Check if model has vision capabilities and if any input has images
                has_vision_module = hasattr(self.model, 'vision_embeddings') and self.model.vision_embeddings is not None
                has_images_in_batch = any(getattr(inp, 'images', None) is not None and len(inp.images) > 0 for inp in llm_inputs)

                if has_vision_module and has_images_in_batch:
                    # Ensure tokenizer has the required special token
                    if hasattr(self.tokenizer, 'special_tokens') and "<|patch|>" in self.tokenizer.special_tokens:
                        patch_token_id = self.tokenizer.special_tokens["<|patch|>"]
                        # Create mask for image patch locations within the initial context
                        image_mask = tokens[:, prev_pos:cur_pos] == patch_token_id
                        image_mask = image_mask.unsqueeze(-1) # Shape: [bsz, seq_len, 1]

                        # Get token embeddings for the initial context
                        h = self.model.tok_embeddings(tokens[:, prev_pos:cur_pos])

                        # Prepare image batch (handle cases with no images for some items)
                        image_batch = [(inp.images if getattr(inp, 'images', None) is not None else []) for inp in llm_inputs]

                        try:
                            # Get vision embeddings using the token embeddings and mask
                            vision_emb = self.model.vision_embeddings(image_batch, image_mask, h)
                            image_embedding = MaskedEmbedding(
                                embedding=vision_emb,
                                mask=image_mask,
                            )
                            print("Processed vision embeddings.")
                        except Exception as e:
                            print(f"Warning: Error processing vision embeddings: {e}. Skipping.")
                            image_embedding = None # Ensure it's None if processing fails
                    else:
                        print("Warning: Vision model detected but '<|patch|>' special token not found in tokenizer. Skipping image processing.")

            # --- Prepare Input for Transformer ---
            xformer_input = TransformerInput(
                tokens=tokens[:, prev_pos:cur_pos], # Input tokens for this step
                tokens_position=prev_pos,          # Starting position for KV cache
                image_embedding=image_embedding,   # Pass image embedding (if any, only on first step)
            )

            # --- Forward Pass ---
            try:
                xformer_output = self.model.forward(xformer_input)
                logits = xformer_output.logits # Shape: [bsz, seq_len_step, vocab_size]
            except Exception as e:
                 print(f"Error during model forward pass at position {cur_pos}: {e}")
                 # Decide how to handle: break, yield error, etc.
                 # For now, let's break the generation for safety.
                 break # Stop generation if forward pass fails

            # --- Process Logits for the Next Token ---
            # We only need the logits for the very last token generated in this step
            current_logits = logits[:, -1, :] # Shape: [bsz, vocab_size]

            # Apply logits processor if provided
            if logits_processor is not None:
                try:
                    # Pass only the necessary parts: current tokens and the last logit slice
                    current_logits = logits_processor(tokens[:, :cur_pos], current_logits)
                except Exception as e:
                     print(f"Warning: Error applying logits_processor: {e}. Using original logits.")


            # --- Sample the Next Token ---
            if temperature > 0:
                # Apply temperature scaling
                probs = torch.softmax(current_logits / temperature, dim=-1)
                # Apply top-p nucleus sampling
                next_token = sample_top_p(probs, top_p) # sample_top_p function defined below
            else:
                # Greedy decoding (temperature == 0)
                next_token = torch.argmax(current_logits, dim=-1)

            next_token = next_token.reshape(-1) # Ensure shape is [bsz]

            # --- Update Tokens Tensor ---
            # Only replace the token if it's in the generation phase (not part of the original prompt)
            # Use the mask calculated *before* filling the current position
            is_prompt_token = input_text_mask[:, cur_pos]
            # Where it was a prompt token, keep the original; otherwise, use the generated token.
            tokens[:, cur_pos] = torch.where(is_prompt_token, tokens[:, cur_pos], next_token)

            # --- Calculate Logprobs (if requested) ---
            if logprobs and token_logprobs is not None:
                 # Calculate log softmax of the logits used for sampling
                 log_probs_for_cur_pos = F.log_softmax(current_logits, dim=-1) # Shape: [bsz, vocab_size]
                 # Gather the log probability of the actually chosen token (for each item in batch)
                 # Need to unsqueeze next_token to use gather properly
                 next_token_logprobs = torch.gather(log_probs_for_cur_pos, dim=-1, index=tokens[:, cur_pos].unsqueeze(-1)) # Use the final token at cur_pos
                 token_logprobs[:, cur_pos] = next_token_logprobs.squeeze(-1) # Store logprob, remove added dim

            # --- Check for End-of-Sequence ---
            # EOS is reached if the token is *generated* (not prompt) AND is a stop token
            is_generated_token = ~is_prompt_token # Negation of the prompt mask
            is_stop_token = torch.isin(tokens[:, cur_pos], stop_tokens)
            # Update eos_reached only for sequences that haven't finished yet
            eos_reached = eos_reached | (is_generated_token & is_stop_token)

            # --- Yield Results for the Current Step ---
            results = []
            for idx in range(bsz):
                 t = tokens[idx, cur_pos]
                 current_logprobs_val = token_logprobs[idx, cur_pos].item() if logprobs and token_logprobs is not None else None
                 # Determine if the token should be ignored by the consumer
                 # Ignore if it was part of the input prompt (and echo is false, handled by source='input')
                 # or if it's padding in a batch where others are still generating.
                 # The primary flag is `finished`. Consumers usually stop processing per-item when finished=True.
                 ignore_token_flag = is_prompt_token[idx].item() and not echo # Ignore prompt tokens if not echoing

                 results.append(
                    GenerationResult(
                        token=t.item(),
                        text=self.tokenizer.decode([t.item()]), # Decode just the single token
                        source="output", # Mark as generated token
                        logprobs=current_logprobs_val,
                        batch_idx=idx,
                        finished=eos_reached[idx].item(), # Has this sequence finished?
                        ignore_token=ignore_token_flag, # Should consumer potentially ignore this? (e.g., prompt part)
                    )
                )
            yield results

            # --- Prepare for Next Iteration ---
            prev_pos = cur_pos # Update position for the next KV cache input

            # --- Check if All Sequences Finished ---
            if torch.all(eos_reached):
                break # Exit loop if all sequences in the batch have generated an EOS token

    def completion(
        self,
        contents: List[RawContent],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Generator[List[GenerationResult], None, None]:
        """
        Generates completions for a batch of raw content inputs.

        Args:
            contents: A list of RawContent objects, each containing text and optional images.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            max_gen_len: Maximum number of new tokens to generate.
            logprobs: Whether to return log probabilities.
            echo: Whether to yield input tokens as well.

        Yields:
            A list of GenerationResult objects for each step of generation.
        """
        try:
            # Encode the raw content into the format expected by the model
            llm_inputs = [self.formatter.encode_content(c) for c in contents]
        except Exception as e:
             print(f"Error encoding content: {e}")
             # Return an empty generator or raise the error
             return
             # raise e

        # Delegate the actual generation process
        yield from self.generate(
            llm_inputs=llm_inputs,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
            logprobs=logprobs,
            echo=echo,
        )
        # The generate method handles the stopping condition (EOS or max_len)

    def chat_completion(
        self,
        messages_batch: List[List[RawMessage]],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False, # Echo is less common in chat completion but supported
    ) -> Generator[List[GenerationResult], None, None]:
        """
        Generates chat completions for a batch of message histories.

        Args:
            messages_batch: A list of message lists. Each inner list represents a conversation history.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            max_gen_len: Maximum number of new tokens to generate for the assistant's response.
            logprobs: Whether to return log probabilities.
            echo: Whether to yield the formatted prompt tokens before the response.

        Yields:
            A list of GenerationResult objects for each step of generation.
        """
        try:
            # Encode the message history into the format expected by the model
            llm_inputs = [self.formatter.encode_dialog_prompt(messages) for messages in messages_batch]
        except Exception as e:
             print(f"Error encoding dialog prompt: {e}")
             # Return an empty generator or raise the error
             return
             # raise e

        # Delegate the actual generation process
        yield from self.generate(
            llm_inputs=llm_inputs,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
            logprobs=logprobs,
            echo=echo,
        )
        # The generate method handles the stopping condition (EOS or max_len)


# --- Helper Function ---
def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor (e.g., shape [bsz, vocab_size]).
                              Assumes probabilities sum to 1 across the last dimension.
        p (float): Probability threshold for top-p sampling (0.0 < p <= 1.0).
                   If p is 0 or 1, it effectively becomes greedy or samples from the full distribution respectively.

    Returns:
        torch.Tensor: Sampled token indices (shape [bsz]).
    """
    if p == 0: # Treat p=0 as greedy sampling
        return torch.argmax(probs, dim=-1)
    if p >= 1.0: # Treat p>=1 as sampling from the full distribution (renormalization handles it)
        p = 1.0

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    # Create mask for values to keep (cumulative probability <= p)
    # Include the first element that *crosses* the threshold p
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0 # Zero out probabilities below the threshold

    # Renormalize the selected probabilities
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    # Sample from the renormalized distribution
    # multinomial expects probabilities, probs_sort now contains the renormalized distribution
    next_token_indices = torch.multinomial(probs_sort, num_samples=1) # Shape [bsz, 1]

    # Map the sampled indices back to the original vocabulary indices
    next_token = torch.gather(probs_idx, dim=-1, index=next_token_indices) # Shape [bsz, 1]

    return next_token.squeeze(-1) # Return shape [bsz]


# --- Example Usage ---
if __name__ == '__main__':

    # --- This is a hypothetical example ---
    # --- You need actual checkpoint files and a tokenizer setup ---
    # --- Replace placeholder paths with your actual file locations ---

    # 1. Define paths and parameters
    #    Ensure these paths point to your downloaded model checkpoint,
    #    params.json, and tokenizer file.
    CHECKPOINT_DIR = "my_transformer_checkpoints" # e.g., "./tinyllama-1.1b-chat-v0.3"
    # TOKENIZER_PATH = "path/to/your/tokenizer.model" # Tokenizer path might be needed for initialization depending on your Tokenizer class
    MAX_SEQ_LEN = 512  # Adjust based on the model and your needs
    MAX_BATCH_SIZE = 1 # Keep batch size small for single inference example

    # 2. Initialize Tokenizer (if required by your implementation)
    #    This step depends heavily on how your Tokenizer class is designed.
    #    If it's a singleton loaded elsewhere or doesn't need explicit init, skip this.
    #    Example:
    #    try:
    #        Tokenizer.initialize(TOKENIZER_PATH) # Hypothetical initialization
    #        print("Tokenizer initialized.")
    #    except Exception as e:
    #        print(f"Failed to initialize tokenizer: {e}")
    #        sys.exit(1) # Exit if tokenizer fails

    # 3. Build the model
    llama_model = None # Initialize variable
    try:
        print("Building model...")
        # The build method now handles getting the tokenizer instance internally
        llama_model = TinyLlama.build(
            ckpt_dir=CHECKPOINT_DIR,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=MAX_BATCH_SIZE,
            # quantization_mode=None, # Optional: e.g., QuantizationMode.int4_mixed
            seed=123,
        )
        print("Model built successfully.")

    except FileNotFoundError as e:
        print(f"\nError: Could not find necessary files.")
        print(f"Please ensure the checkpoint directory '{CHECKPOINT_DIR}' exists and contains:")
        print(f"  - 'params.json'")
        print(f"  - At least one checkpoint file (e.g., 'consolidated.00.pth')")
        # print(f"Also ensure the tokenizer path '{TOKENIZER_PATH}' is correct if needed for initialization.")
        print(f"Details: {e}")
        sys.exit(1) # Exit if model build fails
    except ImportError as e:
         print(f"\nError: Missing import.")
         print(f"Please ensure all required libraries (torch, etc.) and model-specific modules (model.*, chat_format) are installed and accessible.")
         print(f"Details: {e}")
         sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during model build: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Proceed only if model build was successful
    if llama_model:
        # --- Example 4: Chat Completion ---
        try:
            print("\n--- Running Chat Completion Example ---")
            # Prepare input for chat completion
            messages_batch = [
                [
                    # Example using RawMessage (ensure RawMessage class is defined correctly)
                    RawMessage(role="system", content="You are a helpful AI assistant named TinyChat."),
                    RawMessage(role="user", content="What is the capital of France?"),
                ]
                # Add more message lists here for batching > 1
            ]

            # Generate chat completion (stream results)
            print("\nGenerating chat response:")
            full_response = ""
            # Use the chat_completion method of the instantiated model
            for results_list in llama_model.chat_completion(
                messages_batch,
                max_gen_len=50,      # Max new tokens to generate
                temperature=0.7,   # Add some randomness
                top_p=0.9          # Use nucleus sampling
            ):
                # Process results for the first (and only, in this case) batch item
                result = results_list[0] # Assuming batch size is 1
                if not result.ignore_token and not result.finished: # Print generated tokens as they come
                    print(result.text, end="", flush=True)
                    full_response += result.text
                # Stop printing for this item once it's finished
                # The generator will stop yielding for finished items internally

            print("\n--- End of Chat Generation ---")
            # print(f"\nFull Response: {full_response}") # Already printed streamed

        except Exception as e:
            print(f"\nAn error occurred during chat completion: {e}")
            import traceback
            traceback.print_exc()


        # --- Example 5: Standard Completion ---
        try:
            print("\n--- Running Standard Completion Example ---")
            # Prepare input for standard completion
            contents = [
                 # Example using RawContent (ensure RawContent class is defined correctly)
                 RawContent(text="Once upon a time, in a land far, far away, there lived a ")
                 # Add images here if using a vision model and RawContent supports it:
                 # images=[image_data_1, image_data_2]
            ]

            # Generate standard completion (stream results)
            print("\nGenerating standard completion:")
            full_response_completion = ""
            # Use the completion method of the instantiated model
            for results_list in llama_model.completion(
                contents,
                max_gen_len=100,     # Generate more tokens for a story continuation
                temperature=0.8,
                top_p=0.9
            ):
                 result = results_list[0] # Assuming batch size is 1
                 if not result.ignore_token and not result.finished:
                     print(result.text, end="", flush=True)
                     full_response_completion += result.text

            print("\n--- End of Standard Completion ---")
            # print(f"\nFull Response: {full_response_completion}") # Already printed streamed

        except Exception as e:
            print(f"\nAn error occurred during standard completion: {e}")
            import traceback
            traceback.print_exc()

