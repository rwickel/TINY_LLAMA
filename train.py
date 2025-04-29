# train_.py
import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.backends.cuda import sdp_kernel # Keep for checking
from transformers import DataCollatorForLanguageModeling, get_scheduler # Keep get_scheduler

# Version printing (keep as is)
import torchvision
import torchaudio
print("python version:", sys.version)
# print("python version info:", sys.version_info) # Optional verbose
print("torch version:", torch.__version__)
print("cuda version (torch):", torch.version.cuda)
print("torchvision version:", torchvision.__version__)
print("torchaudio version:", torchaudio.__version__)
print("cuda available:", torch.cuda.is_available())

# Custom module imports (ensure paths are correct)
try:
    from model.args import ModelArgs
    from model.model import LLM
    from model.datatypes import TransformerInput
    from model.tokenizer import Tokenizer
    from trainer.config import TrainingConfig
    from trainer.trainer import Trainer
    from trainer.train_datasets import load_tinystories_dataset,default_dataset # Assuming this exists and works
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Ensure model, trainer directories are in PYTHONPATH or relative paths are correct.")
    exit()

if __name__ == '__main__':
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # --- Essential Setup ---
    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()
    # torch.autograd.set_detect_anomaly(True)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True    

    # Note: enable_flash_sdp etc. might require specific torch/CUDA/flash-attn versions
    # Wrap in try-except if causing issues on some setups
    try:
        if torch.cuda.is_available():
             print("Attempting to enable SDP kernels...")
             # torch.backends.cuda.enable_flash_sdp(True) # Might be default or implicit now
             torch.backends.cuda.enable_mem_efficient_sdp(True)
             # torch.backends.cuda.enable_math_sdp(True) # Usually default
             # Check available kernels:
             # with sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True) as V:
             #      print(f"SDP kernel settings: {V}")
             print("SDP kernels enabled (if available).")
        else:
             print("CUDA not available, cannot enable SDP kernels.")
    except Exception as e:
        print(f"Could not configure SDP kernels: {e}")


    # --- Configuration Loading ---
    config = TrainingConfig() # Load defaults

    # --- Overwrite Config with Script Settings ---
    # Hyperparameters (use these to override config defaults)
    config.batch_size = 32 # Adjusted from 64 for potentially smaller GPUs
    config.learning_rate = 3e-4 # Common starting point
    config.epochs = 300
    config.gradient_accumulation_steps = 1 # Default is 1, can be adjusted for larger effective batch size
    # config.gradient_accumulation_steps = 1 # Default is 1
    config.gradient_clipping = 1.0
    config.weight_decay = 0.1
    config.use_amp = torch.cuda.is_available() # Enable AMP only if CUDA is present
    config.resume_from_checkpoint = True # Set True to attempt resuming
    config.save_path = "transformer_checkpoints" # Specific path for this run
    config.debug_mode = True # Enable debug prints in Trainer
    config.debug_print_freq = 10 # Print every 10 steps
    config.lr_scheduler_type = "cosine" # Explicitly set
    config.warmup_ratio = 0.05 # 5% warmup
    # Set max_seq_len used for dataset loading and model args
    MAX_SEQ_LEN = 512

    # --- Device Setup ---
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(config.device)
    print(f"Using device: {device}")
    print(f"Using Automatic Mixed Precision: {config.use_amp}")


    # --- Tokenizer ---
    try:
        # Ensure Tokenizer class path and base_model_name are correct
        tokenizer = Tokenizer(base_model_name="cl100k_base") # Or your specific tokenizer
        config.vocab_size = tokenizer.model.n_vocab # Set vocab size in config
        pad_id = tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else -100 # Get pad_id or use default ignore_index
        if pad_id == -100:
             print("Tokenizer does not have 'pad_id', using default ignore_index -100 for loss.")
        elif pad_id is None:
             print("Warning: Tokenizer 'pad_id' is None, using ignore_index -100 for loss.")
             pad_id = -100
        else:
             print(f"Tokenizer loaded. Vocab size: {config.vocab_size}. Using pad_id: {pad_id}")

    except FileNotFoundError:
         print(f"Error: Tokenizer file not found (check base_model_name or Tokenizer implementation)")
         exit()
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        exit()

    # --- Model Args ---
    # Pass necessary args, ensuring consistency with config and tokenizer
    model_args = ModelArgs(
        dim=512,              # Example size
        n_layers=8,          # Example size
        n_heads=8,            # Needs to divide dim usually (512/8=64)
        n_kv_heads=2,         # Example GQA
        head_dim=64,          # dim / n_heads
        batch_size=config.batch_size, # From training config
        vocab_size=config.vocab_size, # From training config (set by tokenizer)
        max_seq_len=MAX_SEQ_LEN,      # From script variable
        rope_theta=10000.0,   # Common value
        norm_eps=1e-5,
        ffn_dim_multiplier=2, # Optional, example
        device=config.device  # From config
    )

    # --- Instantiate the Model ---
    print("Instantiating Transformer model...")
    model = LLM(model_args)
    model.to(device)

    # --- Compile the Model ---
    # print(f"Attempting to compile model with torch.compile (PyTorch version: {torch.__version__})...")
    # try:
    #     # Using the default mode is a good starting point
    #     model = torch.compile(model)
    #     print("Model successfully compiled.")
    # except Exception as e:
    #     print(f"Warning: Model compilation failed: {e}. Proceeding with eager mode.")
    # -------------------------
        
    # --- Sanity Check ---
    assert tokenizer.model.n_vocab == model_args.vocab_size, \
        "Tokenizer vocab size doesn't match model vocab size"

    # --- Datasets & DataLoaders ---
    try:
        print("Loading datasets...")
        # Adjust max_samples for full run vs debug run
        train_dataset = load_tinystories_dataset(tokenizer, MAX_SEQ_LEN, split="train", max_samples=(config.batch_size * 5000)) # Small sample for test
        valid_dataset = load_tinystories_dataset(tokenizer, MAX_SEQ_LEN, split="validation", max_samples=(config.batch_size * 1)) # Small sample for test
        
        #train_dataset = default_dataset(tokenizer, MAX_SEQ_LEN) # Small sample for test
        #valid_dataset = default_dataset(tokenizer, MAX_SEQ_LEN) # Small sample for test
        
        print(f"Train dataset size (sampled): {len(train_dataset)}")
        print(f"Validation dataset size (sampled): {len(valid_dataset)}")

        if len(train_dataset) == 0 or len(valid_dataset) == 0:
             print("Error: One or both datasets are empty after loading/sampling.")
             exit()

    except FileNotFoundError:
        print("Error: Dataset files not found (check load_tinystories_dataset paths/implementation)")
        exit()
    except Exception as e:
        print(f"Error loading datasets: {e}")
        exit()

    # Data Collator (Optional - Uncomment if load_tinystories_dataset doesn't format batches)
    # print("Using DataCollatorForLanguageModeling.")
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer.model, # Pass the underlying tokenizer model if needed
    #     mlm=False,                 # False for causal LM
    #     return_tensors="pt"
    # )
    # collate_fn = data_collator # Use this in DataLoader below if uncommented
    collate_fn = None # Assume dataset handles batch formatting


    num_cpus = os.cpu_count() or 1
    num_workers = min(4, num_cpus) # Cap workers
    config.num_workers = num_workers # Update config if needed elsewhere
    print(f"Using {num_workers} dataloader workers.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn, # Set to data_collator if using it
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False, # Pin memory only for CUDA
        persistent_workers=True if config.num_workers > 0 else False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size, # Can use same or different batch size for validation
        collate_fn=collate_fn, # Use same collator if needed
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False,
        persistent_workers=True if config.num_workers > 0 else False
    )
    print("DataLoaders created.")

    # --- Optimizer ---
    # Filter parameters that don't require gradients (e.g., frozen embeddings)
    optimizer_grouped_parameters = [
        p for p in model.parameters() if p.requires_grad
    ]
    if len(optimizer_grouped_parameters) < len(list(model.parameters())):
        print("Note: Optimizer will not optimize parameters where requires_grad=False.")

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(0.9, 0.95), # Common values
        eps=1e-8,
        weight_decay=config.weight_decay
    )
    print(f"Optimizer AdamW created with LR={config.learning_rate}, WD={config.weight_decay}")

    # --- Loss Function ---
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    print(f"Loss function: CrossEntropyLoss (ignoring index {pad_id})")

    # --- LR Scheduler ---
    # Calculate total steps and warmup steps here, then create scheduler
    estimated_total_train_steps = (len(train_loader) // config.gradient_accumulation_steps) * config.epochs
    config.warmup_steps = int(estimated_total_train_steps * config.warmup_ratio)
    print(f"Estimated total training steps: {estimated_total_train_steps}")
    print(f"Calculated warmup steps: {config.warmup_steps}")

    lr_scheduler = get_scheduler(
        config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=estimated_total_train_steps
    )
    print(f"LR Scheduler '{config.lr_scheduler_type}' created.")

    # --- Initialize Trainer ---
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,      # Pass criterion
        lr_scheduler=lr_scheduler,# Pass scheduler
        train_loader=train_loader,
        val_loader=None,  # Pass the actual validation loader
        config=config,            # Pass the configured object
        device=device,
        tokenizer=tokenizer,
    )
    print("Trainer initialized.")

    # --- Start Training ---
    print("\nStarting training...")
    trainer.train() # The resume logic is now inside Trainer based on config

    print("\nTraining script finished.")