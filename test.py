import os
import sys
import torch
import torch.nn.functional as F
import math # Needed for sqrt in scaled_dot_product_attention if used directly
import json # Needed for loading params.json
from pathlib import Path # For handling paths
from typing import Tuple, Optional, List, Generator # Added Generator

# --- Assumed Custom Module Imports ---
# Make sure these paths are correct relative to where you run the script,
# or that the 'model' and 'trainer' directories are in your PYTHONPATH.
try:
    # Assuming your model needs ModelArgs for instantiation
    from model.args import ModelArgs
    # Your custom Transformer model class
    from model.model import LLM
    # Input type for your model's forward pass
    from model.datatypes import TransformerInput, TransformerOutput
    # Your custom Tokenizer class
    from model.tokenizer import Tokenizer
    # Import the generation function and helper from generation.py
    # *** Corrected: generate_text is the function to import/use ***
    from generation import generate_text, sample_top_p # Import sample_top_p
    # You might need TrainingConfig if ModelArgs depends on it, but likely not for inference
    # from trainer.config import TrainingConfig
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Ensure 'model', 'trainer', and 'generation.py' are accessible.")
    exit()

# --- Helper Function: sample_top_p is now imported from generation.py ---

# --- Core Generation Function is now imported from generation.py ---

# --- Main Interactive Testing Script ---
if __name__ == '__main__':
    # --- Configuration ---
    # *** USER: MUST SET THESE VALUES ***
    TOKENIZER_NAME = "cl100k_base" # Or the specific name/path used during training
    CHECKPOINT_DIR = 'my_transformer_checkpoints' # Directory where checkpoints are saved

    # --- Model arguments will be loaded from params.json ---

    # Generation parameters
    MAX_NEW_TOKENS = 150
    TEMPERATURE = 0.7
    TOP_P = 0.9
    # --- End Configuration ---

    # --- Checkpoint Finding ---
    LATEST_CHECKPOINT_FILENAME = None
    CHECKPOINT_PATH = None
    PARAMS_PATH = None
    try:
        checkpoint_dir_path = Path(CHECKPOINT_DIR)
        if not checkpoint_dir_path.is_dir():
             raise FileNotFoundError(f"Checkpoint directory not found: {CHECKPOINT_DIR}")

        # Find params.json first
        PARAMS_PATH = checkpoint_dir_path / "params.json"
        if not PARAMS_PATH.is_file():
             raise FileNotFoundError(f"params.json not found in checkpoint directory: {CHECKPOINT_DIR}")

        # Find checkpoint files
        checkpoints = sorted([f for f in checkpoint_dir_path.glob('checkpoint_*.pt')]) # Use glob and sort
        if not checkpoints:
            raise FileNotFoundError("No .pt checkpoint files starting with 'checkpoint_' found in directory.")

        # Sort checkpoints (example: assumes format like checkpoint_epoch_XXX_step_YYY_loss_ZZZ.pt)
        def get_sort_key(filepath: Path):
            filename = filepath.name
            try:
                 # Extract epoch and step, handle potential errors gracefully
                 epoch_str = filename.split('_')[2]
                 step_str = filename.split('_')[4]
                 epoch = int(epoch_str)
                 step = int(step_str)
                 return epoch, step
            except (IndexError, ValueError):
                 # If filename format is unexpected, return a value that sorts it last
                 print(f"Warning: Could not parse epoch/step from filename: {filename}. Placing it last.")
                 return (-1, -1) # Sort unparseable names last

        checkpoints.sort(key=get_sort_key, reverse=True) # Sort by epoch, then step (descending)

        LATEST_CHECKPOINT_FILENAME = checkpoints[0].name
        CHECKPOINT_PATH = checkpoints[0] # Use the Path object directly
        print(f"Found params.json: {PARAMS_PATH}")
        print(f"Using latest checkpoint: {CHECKPOINT_PATH}")

    except FileNotFoundError as e:
        print(f"Error finding required files in '{CHECKPOINT_DIR}': {e}")
        exit()
    except Exception as e:
        print(f"Error during checkpoint/params finding in '{CHECKPOINT_DIR}': {e}")
        exit()


    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Tokenizer
    try:
        print(f"Loading tokenizer: {TOKENIZER_NAME}")
        tokenizer = Tokenizer(base_model_name=TOKENIZER_NAME)
        # Ensure tokenizer has n_vocab attribute after loading
        if not hasattr(tokenizer, 'model') or not hasattr(tokenizer.model, 'n_vocab'):
             raise AttributeError("Tokenizer object or its 'model' attribute missing 'n_vocab'.")
        print(f"Tokenizer loaded. Vocab size: {tokenizer.model.n_vocab}")
    except Exception as e:
        print(f"Error loading tokenizer '{TOKENIZER_NAME}': {e}")
        exit()

    # --- Load Model Arguments from params.json ---
    try:
        print(f"Loading model arguments from: {PARAMS_PATH}")
        with open(PARAMS_PATH, "r") as f:
            params = json.load(f)
            print("Successfully loaded params.json.")

        # Instantiate ModelArgs using loaded params
        # Ensure required args like max_seq_len are present or handled
        # max_seq_len might be defined during training config, not necessarily saved in params.json
        # If it's crucial for ModelArgs init, ensure it's in params or handle default/override
        if 'max_seq_len' not in params:
             print("Warning: 'max_seq_len' not found in params.json. Ensure ModelArgs handles this or set a default.")
             # Example: Set a default if ModelArgs requires it
             # params['max_seq_len'] = 512 # Or get from a config object if available

        model_args: ModelArgs = ModelArgs(**params)

        # Set vocab_size in model_args AFTER loading from json
        # This is usually runtime/environment specific, not architecture specific
        model_args.vocab_size = tokenizer.model.n_vocab

        # *** REMOVED line causing the error ***
        # model_args.device = device # Cannot set ClassVar on instance

        print("Model arguments configured:")
        # Pretty print the args
        try:
            # Use model_dump if available (Pydantic v2+) else __dict__
            if hasattr(model_args, 'model_dump'):
                 print(json.dumps(model_args.model_dump(), indent=2))
            elif hasattr(model_args, '__dict__'):
                 # Convert potentially complex objects (like device) to str for printing
                 # Exclude the 'device' ClassVar if it exists in __dict__
                 printable_args = {k: str(v) if isinstance(v, torch.device) else v
                                   for k, v in model_args.__dict__.items() if k != 'device'}
                 print(json.dumps(printable_args, indent=2))
            else: # Fallback for objects without __dict__ (less common)
                 print(vars(model_args))

        except TypeError as e: # Handle potential non-serializable types if ModelArgs is complex
            print(f"Could not serialize all model args for printing: {e}")
            print(vars(model_args)) # Print raw vars as fallback


    except FileNotFoundError:
        print(f"Error: params.json not found at {PARAMS_PATH}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {PARAMS_PATH}")
        exit()
    except TypeError as e:
        print(f"Error initializing ModelArgs with parameters from params.json: {e}")
        print("Ensure params.json content matches ModelArgs fields.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred loading model args: {e}")
        exit()


    # Instantiate Model
    print("Instantiating model...")
    # Pass the configured model_args object
    model = LLM(model_args)
    model.to(device) # Move model structure to device before loading state_dict

    # Load Checkpoint
    try:
        print(f"Loading checkpoint state dictionary: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device) # Load directly to the target device

        # Handle potential DataParallel or DistributedDataParallel wrappers
        state_dict = checkpoint.get('model_state_dict', None)
        if state_dict is None:
             # Try loading older format if 'model_state_dict' is missing
             print("Warning: 'model_state_dict' key not found. Attempting to load the entire checkpoint as state_dict.")
             state_dict = checkpoint

        # Remove `module.` prefix if saved with DataParallel/DDP
        unwrapped_state_dict = {}
        needs_unwrapping = any(k.startswith('module.') for k in state_dict.keys())
        if needs_unwrapping:
            print("Unwrapping 'module.' prefix from state_dict keys...")
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    unwrapped_state_dict[k[len('module.'):]] = v
                else:
                    # If some keys have prefix and others don't, it might be an issue
                    unwrapped_state_dict[k] = v
        else:
            unwrapped_state_dict = state_dict

        # Load the state dict (use strict=True for better error checking during inference setup)
        load_result = model.load_state_dict(unwrapped_state_dict, strict=True)
        print(f"Model state dictionary load result: {load_result}")

        # Optional: Load optimizer/epoch/step if needed for other purposes, but not for generation
        epoch = checkpoint.get('epoch', 'N/A')
        step = checkpoint.get('global_step', 'N/A')
        print(f"Checkpoint info - Epoch: {epoch}, Step: {step}")

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        exit()
    except KeyError as e:
         print(f"Error: Checkpoint missing required key: {e}")
         exit()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # Print detailed traceback for debugging loading errors
        import traceback
        traceback.print_exc()
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

            # Generate text using the imported function
            # *** Corrected function name from 'generate' to 'generate_text' ***
            full_response, attn_weights = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                device=device,
                return_attn_weights=False # Set True only if you want to process weights
            )
            # The generate_text function now handles the streaming print internally

            # If attn_weights were requested and returned, you can process them here:
            if attn_weights is not None:
                 print(f"\nReceived attention weights tensor with shape: {attn_weights.shape}")
                 # Add code here to visualize or analyze attn_weights

            print("-" * 50) # Separator

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Exiting.")
            break
        except Exception as e:
            print(f"\nAn error occurred during generation: {e}")
            # Continue the loop
