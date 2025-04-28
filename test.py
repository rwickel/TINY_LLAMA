import os
import sys
import torch
import torch.nn.functional as F

# --- Assumed Custom Module Imports ---
# Make sure these paths are correct relative to where you run the script,
# or that the 'model' and 'trainer' directories are in your PYTHONPATH.
try:
    from model.args import ModelArgs # Assuming your model needs this
    from model.model import LLM     # Your custom Transformer model class
    from model.datatypes import TransformerInput # Input type for your model's forward pass
    from model.tokenizer import Tokenizer # Your custom Tokenizer class
    # You might need TrainingConfig if ModelArgs depends on it, but likely not for inference
    # from trainer.config import TrainingConfig
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Ensure 'model', 'trainer' directories are accessible.")
    exit()

# --- Helper Function: sample_top_p (Copied from your generation code) ---
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.
    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.
    Returns:
        torch.Tensor: Sampled token indices.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    # Ensure no division by zero if all probabilities become zero after masking
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True).clamp(min=1e-8))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

# --- Core Generation Function ---
@torch.inference_mode()
def generate_text(
    model: LLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda" # Or torch.device object
):
    """
    Generates text token by token based on a prompt using the provided model and tokenizer.

    Args:
        model: The loaded LLM model instance.
        tokenizer: The loaded Tokenizer instance.
        prompt: The input string prompt.
        max_new_tokens: The maximum number of new tokens to generate.
        temperature: Sampling temperature. 0 means greedy decoding.
        top_p: Nucleus sampling probability.
        device: The device to run generation on ('cuda' or 'cpu').

    Yields:
        str: The decoded text of each generated token.
    """
    model.eval() # Ensure model is in evaluation mode
    model.to(device)

    # --- Prepare Input ---
    # Encode the prompt. Assuming bos=True is needed for your model.
    # Adjust eos=False/True based on how your model was trained.
    prompt_tokens = tokenizer.encode(prompt, bos=True, eos=False)
    prompt_tokens_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    bsz, prompt_len = prompt_tokens_tensor.shape
    assert bsz == 1, "This function only supports batch size 1 for interactive use"

    # --- Setup Generation ---
    if not hasattr(model, 'args') or not hasattr(model.args, 'max_seq_len'):
         raise AttributeError("Model object must have an 'args' attribute with 'max_seq_len' defined.")
    max_seq_len = model.args.max_seq_len
    total_len = min(max_seq_len, prompt_len + max_new_tokens)

    if prompt_len >= total_len:
        print(f"Warning: Prompt length ({prompt_len}) is already >= max sequence length ({max_seq_len}) or max_new_tokens limit. No new tokens will be generated.")
        # Optionally yield the prompt back or do nothing
        # decoded_prompt = tokenizer.decode(prompt_tokens)
        # yield decoded_prompt
        return

    # Use the tokenizer's pad_id. Handle potential missing attribute.
    pad_id = getattr(tokenizer, 'pad_id', -1) # Use -1 or another value if pad_id isn't standard
    if pad_id == -1:
        print("Warning: Tokenizer does not have 'pad_id'. Using -1 as placeholder.")

    # Prepare buffer for all tokens (prompt + generated)
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
    tokens[0, :prompt_len] = prompt_tokens_tensor[0] # Fill with prompt tokens

    # Get stop tokens from the tokenizer, if available
    stop_tokens = []
    if hasattr(tokenizer, 'stop_tokens'):
       stop_tokens = torch.tensor(tokenizer.stop_tokens, device=device)

    # --- Generation Loop (Token by Token) ---
    prev_pos = 0 # For KV cache handling in the model's forward pass
    for cur_pos in range(prompt_len, total_len):
        # Prepare input for the model's forward pass (incremental decoding)
        # The model internally uses the KV cache based on tokens_position
        xformer_input = TransformerInput(
            tokens=tokens[:, prev_pos:cur_pos], # Input tokens for this step
            tokens_position=prev_pos,          # Starting position for KV cache indexing
            image_embedding=None               # Assuming no image input for text generation
        )

        # Get logits from the model
        try:
            xformer_output = model.forward(xformer_input)
            logits = xformer_output.logits[:, -1, :] # Get logits for the very last token position
        except Exception as e:
            print(f"\nError during model forward pass at position {cur_pos}: {e}")
            print(f"Input tokens shape: {xformer_input.tokens.shape}")
            print(f"Tokens position: {xformer_input.tokens_position}")
            break # Stop generation on error

        # Sample the next token
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p) # Shape: [1, 1]
        else:
            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1, keepdim=True) # Shape: [1, 1]

        next_token = next_token.reshape(-1) # Reshape to [1] for easier handling

        # Add the sampled token to our buffer
        tokens[0, cur_pos] = next_token

        # Check if the generated token is a stop token
        # Ensure stop_tokens is a tensor and not empty before checking
        if len(stop_tokens) > 0 and torch.isin(next_token, stop_tokens):
            # Optional: Decode and yield the stop token itself if needed
            # decoded_token = tokenizer.decode(next_token.tolist())
            # yield decoded_token + " [EOS]" # Indicate stop
            break # Stop generation

        # Decode the single generated token and yield it for streaming output
        decoded_token = tokenizer.decode(next_token.tolist())
        yield decoded_token

        # Update prev_pos for the next iteration's KV cache window
        prev_pos = cur_pos

# --- Main Interactive Testing Script ---
if __name__ == '__main__':
    # --- Configuration ---
    # *** USER: MUST SET THESE VALUES ***
    TOKENIZER_NAME = "cl100k_base" # Or the specific name/path used during training
    CHECKPOINT_DIR = 'my_transformer_checkpoints' # Directory where checkpoints are saved
    # Find the latest checkpoint or specify one manually
    # Example: Find the latest checkpoint based on epoch/step
    try:
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
        if not checkpoints:
            raise FileNotFoundError("No .pt checkpoint files found in directory.")
        # Sort checkpoints (example: assumes format like checkpoint_epoch_XXX_step_YYY_loss_ZZZ.pt)
        checkpoints.sort(key=lambda x: int(x.split('_')[2]), reverse=True) # Sort by epoch desc
        LATEST_CHECKPOINT_FILENAME = checkpoints[0]
        CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, LATEST_CHECKPOINT_FILENAME)
        print(f"Using latest checkpoint: {CHECKPOINT_PATH}")
    except (FileNotFoundError, IndexError, ValueError, TypeError) as e:
         print(f"Error finding latest checkpoint in '{CHECKPOINT_DIR}': {e}")
         # *** USER: Manually set the path if auto-detection fails ***
         CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_437_step_206880_loss_2.4620.pt') # << SET MANUALLY IF NEEDED
         print(f"Falling back to manual checkpoint path: {CHECKPOINT_PATH}")
         if not os.path.exists(CHECKPOINT_PATH):
              print(f"Error: Manual checkpoint path does not exist: {CHECKPOINT_PATH}")
              exit()


    # Model arguments (MUST match the arguments used during training for the loaded checkpoint)
    # *** USER: MUST SET THESE VALUES TO MATCH TRAINING ***
    model_args = ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2, # Make sure this matches training
        # head_dim will likely be calculated inside ModelArgs or LLM based on dim and n_heads
        # batch_size is not needed for inference args usually
        # vocab_size will be set after loading tokenizer
        max_seq_len=512, # Must match training
        rope_theta=10000.0,
        norm_eps=1e-5,
        ffn_dim_multiplier=2, # Make sure this matches training
        # device will be set below
    )
    # Generation parameters
    MAX_NEW_TOKENS = 150
    TEMPERATURE = 0.7
    TOP_P = 0.9
    # --- End Configuration ---

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Tokenizer
    try:
        print(f"Loading tokenizer: {TOKENIZER_NAME}")
        tokenizer = Tokenizer(base_model_name=TOKENIZER_NAME)
        # Set vocab_size in model_args based on loaded tokenizer
        model_args.vocab_size = tokenizer.model.n_vocab
        print(f"Tokenizer loaded. Vocab size: {model_args.vocab_size}")
        # Add device to model args AFTER setting vocab size
        model_args.device = device
    except Exception as e:
        print(f"Error loading tokenizer '{TOKENIZER_NAME}': {e}")
        exit()

    # Instantiate Model
    print("Instantiating model...")
    model = LLM(model_args)
    model.to(device) # Move model structure to device before loading state_dict

    # Load Checkpoint
    try:
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device) # Load directly to the target device

        # Handle potential DataParallel or DistributedDataParallel wrappers
        state_dict = checkpoint['model_state_dict']
        # Remove `module.` prefix if saved with DataParallel/DDP
        unwrapped_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                unwrapped_state_dict[k[len('module.'):]] = v
            else:
                unwrapped_state_dict[k] = v

        model.load_state_dict(unwrapped_state_dict)
        print("Model state dictionary loaded successfully.")

        # Optional: Load optimizer/epoch/step if needed for other purposes, but not for generation
        # epoch = checkpoint.get('epoch', 0)
        # step = checkpoint.get('global_step', 0)
        # print(f"Checkpoint from Epoch: {epoch}, Step: {step}")

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        exit()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit()

    # --- Interactive Loop ---
    print("-" * 50)
    print("Model loaded. Type your prompt. Type 'exit' or 'quit' to end.")
    print("-" * 50)

    while True:
        try:
            prompt = input("You: ")
            if prompt.strip().lower() in ["exit", "quit"]:
                print("👋 Exiting.")
                break
            if not prompt.strip():
                continue

            print("Model: ", end="", flush=True) # Print "Model: " prefix

            # Generate text using the function, streaming output
            full_response = ""
            for token_text in generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                device=device
            ):
                print(token_text, end="", flush=True) # Print each token without newline
                full_response += token_text

            print() # Add a newline after the full response is generated
            print("-" * 50) # Separator

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Exiting.")
            break
        except Exception as e:
            print(f"\nAn error occurred during generation: {e}")
            # Continue the loop