# Custom Transformer Training Framework

## Overview

This package provides a structured framework for training a custom Transformer decoder model (`DecoderLM`) using PyTorch. It leverages the Hugging Face `datasets` and `transformers` libraries for efficient data handling and tokenization. The framework is designed to be modular, allowing for easier configuration and modification.

Key features include:
* Integration with Hugging Face `datasets`.
* Automatic dataset loading, tokenization, and batching.
* A configurable `Trainer` class managing the training loop.
* Support for Mixed Precision training (`torch.cuda.amp`) via `GradScaler`.
* Cosine learning rate scheduling with linear warmup.
* Checkpointing (saving the latest and best model based on validation loss).
* Resuming training from the latest checkpoint.
* Saving model configuration and tokenizer state for easy reloading.
* Progress bars using `tqdm`.

## Directory Structure

The framework assumes the following project structure:
├── model/
│   ├── config.py       # Your existing TransformerConfig, get_model_config, check_device
│   ├── layers.py       # Your existing layer implementations
│   ├── model.py        # Your existing DecoderLM implementation
│   └── positional_encoding.py # Your existing positional encoding
├── trainer/
│   ├── init.py           # (Package initialization)
│   ├── config.py         # TrainingConfig dataclass
│   ├── squad_data.py     # Data loading and preparation
│   ├── trainer.py        # Trainer class
│   
└── train.py              # Training script


## Core Components (`trainer/`)

* **`config.py`:** Defines the `TrainingConfig` dataclass for all training hyperparameters (learning rate, batch size, epochs, paths, etc.).
* **`squad_data.py`:** Contains functions (`create_dataloaders`) to fetch datasets from the Hugging Face Hub, split them into train/validation sets, tokenize the text, and create `DataLoader` instances using `DataCollatorForLanguageModeling`.
* **`trainer.py`:** Implements the `Trainer` class, which manages:
    * Epoch and step iteration
    * Learning rate scheduling
    * Mixed precision (AMP)
    * Gradient accumulation
    * Optimization steps
    * Validation loops
    * Checkpointing (`latest_checkpoint.pt`, `best_model.pt`)
    * Training resumption
* **`utils.py`:** Provides utility functions, including the `get_lr` learning rate scheduler and functions to load models and tokenizers.

## Training Process

The `Trainer` class in `trainer/trainer.py` orchestrates the training loop. Here's a breakdown of the key steps involved in each training iteration:

1.  **Batch Retrieval:** The `Trainer` fetches a batch of data from the `train_loader`. This batch typically contains `input_ids`, `attention_mask`, and `labels`.

2.  **Data Transfer:** The input data is moved to the specified device (CPU or GPU).

3.  **Forward Pass (with AMP):**
    * If mixed precision training is enabled (`TrainingConfig.use_mixed_precision`), the forward pass is executed within a `torch.cuda.amp.autocast` context. This allows PyTorch to automatically use lower precision (float16) for compatible operations, leading to faster training and reduced memory consumption.
    * The model receives the input data and produces `logits` (predicted token probabilities) and `loss`.

4.  **Backward Pass (with Gradient Scaling):**
    * The calculated `loss` is scaled using `torch.cuda.amp.GradScaler`. This scaling helps prevent underflow issues that can occur with lower precision.
    * The scaled loss is used to compute gradients via `loss.backward()`.

5.  **Gradient Clipping (Optional):**
    * If enabled in `TrainingConfig`, gradients are clipped to a maximum norm to prevent exploding gradients.

6.  **Optimization Step:**
    * The gradients are unscaled using `scaler.unscale_(optimizer)`.
    * The optimizer updates the model's parameters based on the calculated gradients (`optimizer.step()`).

7.  **Learning Rate Update:**
    * The learning rate scheduler (`lr_scheduler`) updates the learning rate based on the current training step.

8.  **Gradient Reset:**
    * The gradients are reset to zero (`optimizer.zero_grad(set_to_none=True)`) to prepare for the next iteration.

9.  **Progress Tracking:**
    * The current loss and learning rate are logged, and the progress bar is updated.

10. **Validation (Periodic):**
    * After a certain number of steps (or at the end of each epoch), the model is evaluated on the `val_loader` to assess its generalization performance.

11. **Checkpointing (Periodic):**
    * The model's state, optimizer state, scheduler state, and training configuration are saved to disk.
    * The best model (based on validation loss) is also saved.

12. **Training Resumption:**
    * If training is interrupted, it can be resumed from the latest saved checkpoint.
