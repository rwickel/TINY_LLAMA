import os
import random
import numpy as np
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling

# --- Custom Model Imports ---
from model.model import DecoderLM

# --- Trainer Imports ---
from trainer.config import TrainingConfig
from trainer.trainer import Trainer
from trainer.squad_data import SQuADDataset  # Renamed to avoid shadowing
from trainer.train_datasets import default_dataset, load_tinystories_dataset


def set_seed(seed_value):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

if __name__ == "__main__":    

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    
    torch.backends.cuda.enable_flash_sdp(True)  # Enable FlashAttention-2
    torch.backends.cuda.enable_mem_efficient_sdp(True)  # Fallback option
    torch.backends.cuda.enable_math_sdp(True)  # Baseline implementation

     

    # --- Configuration ---
    config = TrainingConfig()
    config.device = device  # Update config with actual device
    config.resume_from_checkpoint = True  # Set to True if you want to resume
    config.dataset_name = "tiny_stories"  # Reflecting the change
    config.tokenizer_name_or_path = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # --- Setup ---
    #set_seed(config.seed)
    os.makedirs(config.save_path, exist_ok=True)

    # --- Define the specific model name ---
    emb_model = config.tokenizer_name_or_path  # Use the configured tokenizer name
    print(f"Using '{emb_model}' to derive model config via get_model_config.")

    # --- Tokenizer ---
    print(f"Loading tokenizer: {emb_model}")
    tokenizer = AutoTokenizer.from_pretrained(emb_model)

    # Handle special tokens
    tokenizer.add_special_tokens({
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",
        "additional_special_tokens": [
            "<DOC>", "<ENDDOC>",
            "<CTX>", "</CTX>",
            "<Q>", "</Q>",
            "<A>", "</A>",
            "<SYSTEM>", "</SYSTEM>", "<USER>", "</USER>", "<ASSISTANT>", "</ASSISTANT>",
        ]
    })

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # --- Model Config & Initialization ---
    print(f"Loading base AutoConfig: {emb_model}")
    base_config_obj = AutoConfig.from_pretrained(emb_model)

    print("Calling get_model_config...")
    model_config = get_model_config(
        base_config=base_config_obj,
        tokenizer=tokenizer,
        device=device
    )
    model_config.shift_labels = False 

    config.max_seq_length = model_config.block_size  # Set max_seq_length in config

    if not isinstance(model_config, TransformerConfig):
        raise TypeError(
            f"get_model_config must return an instance of TransformerConfig, but got {type(model_config)}")

    
    print(f"Initializing model with config from get_model_config: {model_config}")
    model = DecoderLM(model_config).to(device)

    print(f"Model initialized on device: {next(model.parameters()).device}")
    print(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M trainable parameters.")

    # --- Data Preparation --- Simplified Call ---
    print(f"Preparing data Dataset...")
    #train_dataset = SQuADDataset(tokenizer, config, split="train", max_samples=1)
    train_dataset = load_tinystories_dataset(tokenizer, config, split="train", max_samples= (config.batch_size *4))  # Use the new function
    #train_dataset = create_default_data(tokenizer, config, split="train", max_samples=100)
    #val_dataset   = SQuADDataset(tokenizer, config, split="validation", max_samples=100) # not used see


    # --- Subsetting (applied AFTER tokenization) ---
    if 0.0 < config.train_data_subset_fraction < 1.0:
        num_train_samples = len(train_dataset)
        subset_size = math.ceil(num_train_samples * config.train_data_subset_fraction)
        print(
            f"Using a subset of the training data: {subset_size} samples ({config.train_data_subset_fraction * 100:.1f}%)")
        train_dataset = torch.utils.data.Subset(train_dataset, range(subset_size))  # Use Subset
    elif config.train_data_subset_fraction >= 1.0:
        print("Using the full training dataset.")
    else:
        raise ValueError("train_data_subset_fraction must be between 0.0 and 1.0 (or >= 1.0 to use all data)")

    # --- DataLoader Creation ---
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )

    train_loader = DataLoader(
        train_dataset,  # Use the tokenized data
        batch_size=config.batch_size,
        #collate_fn=data_collator,  # Collator handles final padding within the batch
        shuffle=False,  # Shuffle for training
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True  # Maintains worker pools
    )
    
    # --- Optimizer ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,  # Initial LR, scheduler will adjust
        weight_decay=config.weight_decay
    )

    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=None,
        config=config,
        device=device,
        tokenizer=tokenizer,
        lr_scheduler_type="cosine"
    )

    # --- Start Training ---
    trainer.train(config.resume_from_checkpoint)