# trainer/config.py
from dataclasses import dataclass, field
import torch
import os

# trainer/config.py
from dataclasses import dataclass, field
import torch
import os

@dataclass
class TrainingConfig:    

    # Resuming
    resume_from_checkpoint: bool = True # Default to False, can be overridden

    # Training Hyperparameters
    epochs: int = 300 # Sensible default
    batch_size: int = 64 # Per device batch size - Default, override in main script
    gradient_accumulation_steps: int = 1 # Effective batch size = batch_size * accumulation_steps
    learning_rate: float = 1e-4 # Default, override in main script
    weight_decay: float = 0.1 # Common value
    gradient_clipping: float = 1.0 # Common value

    # LR Schedule
    lr_scheduler_type: str = "cosine" # Default scheduler type
    decay_lr: bool = True # Usually True for schedulers like cosine
    warmup_ratio: float = 0.05 # % of total steps for warmup
    # min_lr_ratio: float = 0.1 # min_lr = base_learning_rate * min_lr_ratio (Handled by transformers scheduler)

    save_interval : int = 1 # Save checkpoint every n epochs
    
    # Technical
    use_amp: bool = True # Default to True if CUDA available, check in main script
    seed: int = 42

    # Logging & Saving
    log_interval: int = 10 # Log training loss frequency
    eval_interval: int = 50 # Evaluate frequency (in steps)
    save_path: str = "checkpoints" # Directory to save checkpoints
    # checkpoint_filename_latest: str = "latest_checkpoint.pt" # Handled by Trainer naming convention
    # checkpoint_filename_best: str = "best_model.pt" # Requires tracking best loss

    # Dataloader
    # Set num_workers based on CPUs available in main script
    num_workers: int = 6 # Sensible default

    # Debugging
    debug_mode: bool = True # Default to False
    debug_print_freq: int = 10 # Frequency for debug prints if enabled

    # --- Fields to be calculated/set externally ---
    device: str = "cpu" # Will be set in main script
    # total_train_steps: int = 0 # Calculated in main script
    warmup_steps: int = 0 # Calculated in main script based on ratio
    # min_lr: float = 0.0 # Handled by scheduler
    vocab_size: int = 0 # Set after tokenizer is loaded