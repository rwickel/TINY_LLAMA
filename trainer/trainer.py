import os
import re
import time
import datetime
import torch
import torch.nn as nn # Keep for type hinting if desired, but not strictly needed for creation
from tqdm import tqdm
# Removed: from transformers import get_scheduler (no longer created internally)
from trainer.config import TrainingConfig # Assuming TrainingConfig defines relevant params
from model.datatypes import TransformerInput # Assuming this is the correct import for your model
import logging  
import dataclasses 
from model.model import LLM # Assuming LLM is the model class
from model.tokenizer import Tokenizer # Assuming Tokenizer is the tokenizer class
import json

# Setup basic logging (optional, but good practice)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="training.log", # Specify the file path
    filemode='a' # 'a' to append to the file, 'w' to overwrite each time
)
logger = logging.getLogger(__name__)

# --- Trainer Class ---
class Trainer:
    """
    Manages the training and evaluation loop for a transformer model.

    Handles epoch iteration, batch processing, forward/backward passes,
    optimizer and learning rate scheduler steps, automatic mixed precision (AMP),
    gradient clipping, evaluation, and checkpoint saving/loading.
    Accepts pre-configured criterion (loss function) and lr_scheduler.
    Saves model architectural parameters to params.json in the checkpoint directory.
    """
    def __init__(self, model: LLM, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, lr_scheduler, train_loader, val_loader, config: TrainingConfig, device: torch.device, tokenizer: Tokenizer):
        """
        Initializes the Trainer instance.

        Args:
            model (torch.nn.Module): The model to be trained (expected to have an 'args' attribute).
            optimizer (torch.optim.Optimizer): The optimizer for training.
            criterion (torch.nn.Module): The loss function (e.g., nn.CrossEntropyLoss).
            lr_scheduler: The learning rate scheduler instance.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
            config (TrainingConfig): Configuration object containing training parameters
                (epochs, gradient_clipping, use_amp, save_path, resume_from_checkpoint etc.).
            device (torch.device): The device (CPU or CUDA) to run training on.
            tokenizer: The tokenizer used (potentially for debugging/logging).
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion # Expects CrossEntropyLoss or similar with ignore_index
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.tokenizer = tokenizer # Keep tokenizer for potential debug prints
        self.global_step = 0
        self.start_epoch = 0

        # --- AMP Setup ---
        self.use_amp = getattr(config, 'use_amp', False) # Default to False if not specified
        self.pt_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else (torch.float16 if self.use_amp else torch.float32)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        logger.info(f"Automatic Mixed Precision (AMP) {'enabled' if self.use_amp else 'disabled'} with dtype {self.pt_dtype}.")

        # --- Gradient Clipping Setup ---
        self.max_grad_norm = getattr(config, 'gradient_clipping', None) # Use None if not present or <= 0
        if self.max_grad_norm and self.max_grad_norm <= 0:
             self.max_grad_norm = None # Treat non-positive values as disabled
        logger.info(f"Gradient clipping {'enabled' if self.max_grad_norm else 'disabled'}" + (f" with max_norm: {self.max_grad_norm}" if self.max_grad_norm else ""))

        # --- Checkpoint Loading ---
        self.resume_enabled = getattr(config, 'resume_from_checkpoint', False)
        self.save_path_defined = hasattr(config, 'save_path') and config.save_path
        if self.resume_enabled and self.save_path_defined:
             self._load_checkpoint()
        elif self.resume_enabled:
             logger.warning("resume_from_checkpoint is True, but config.save_path is not defined. Cannot load checkpoint.")
        else:
            logger.info("Starting training from scratch (or resume_from_checkpoint is False).")


    def _save_checkpoint(self, epoch, loss_value):
        """
        Saves the current training state to a .pt file and the model's
        architectural parameters to params.json in the save directory.

        The .pt file contains: model weights, optimizer state, scheduler state,
        scaler state (if AMP is used), epoch number, and global step.

        Args:
            epoch (int): The current epoch number (0-indexed).
            loss_value (float): The last loss value recorded (used for filename).
        """
        save_path = getattr(self.config, 'save_path', None)
        if not save_path:
            logger.warning("config.save_path not defined. Cannot save checkpoint.")
            return

        # Ensure the save directory exists
        try:
            os.makedirs(save_path, exist_ok=True)
        except OSError as e:
            logger.error(f"Error creating save directory {save_path}: {e}")
            return # Cannot save if directory cannot be created

        # --- Save Model Parameters to params.json ---
        model_args_saved = False
        if hasattr(self.model, 'args') and self.model.args is not None:
            try:
                model_args_obj = self.model.args
                model_args_dict = None

                # Try converting to dict using dataclasses.asdict or vars()
                if dataclasses.is_dataclass(model_args_obj):
                    model_args_dict = dataclasses.asdict(model_args_obj)
                elif hasattr(model_args_obj, 'model_dump'): # Pydantic V2+
                    model_args_dict = model_args_obj.model_dump(mode='json') # Use json mode for better serialization
                elif hasattr(model_args_obj, 'dict'): # Pydantic V1
                     model_args_dict = model_args_obj.dict()
                elif hasattr(model_args_obj, '__dict__'):
                    model_args_dict = vars(model_args_obj).copy() # Use copy
                else:
                    logger.warning(f"Cannot serialize self.model.args of type {type(model_args_obj)}. Needs to be a dataclass, Pydantic model, or have __dict__.")

                # Sanitize potentially non-serializable types (like torch.device) if conversion succeeded
                if isinstance(model_args_dict, dict):
                    # Remove runtime/non-architectural args before saving to params.json
                    runtime_keys = ['max_seq_len', 'max_batch_size', 'device'] # Keys not part of core architecture
                    for key in runtime_keys:
                        model_args_dict.pop(key, None) # Remove if exists

                    # Convert remaining complex types (example: torch.device)
                    for key, value in model_args_dict.items():
                        if isinstance(value, torch.device):
                            model_args_dict[key] = str(value)
                        # Add more conversions here if needed

                    # Save the processed dictionary to params.json
                    params_path = os.path.join(save_path, "params.json")
                    try:
                        with open(params_path, 'w') as f:
                            json.dump(model_args_dict, f, indent=2)
                        model_args_saved = True
                        # Log only once or periodically if needed
                        # logger.info(f"Model parameters saved to {params_path}")
                    except TypeError as e:
                         logger.error(f"Error serializing model args to JSON: {e}. Check for non-serializable types in model args.", exc_info=True)
                    except IOError as e:
                         logger.error(f"Error writing params.json to {params_path}: {e}", exc_info=True)

            except Exception as e:
                logger.warning(f"Could not serialize or save self.model.args to params.json: {e}.", exc_info=True)
        else:
            logger.warning("self.model does not have a valid 'args' attribute. Model parameters (params.json) will not be saved.")
        # --- End Save Model Parameters ---


        # --- Save Training State to .pt file ---
        # Format the loss value for the filename safely
        loss_str = "nan"
        if isinstance(loss_value, (int, float)) and loss_value == loss_value and abs(loss_value) != float('inf'):
             loss_str = f"{loss_value:.4f}"

        checkpoint_filename = f"checkpoint_epoch_{epoch + 1}_step_{self.global_step}_loss_{loss_str}.pt"
        checkpoint_path = os.path.join(save_path, checkpoint_filename)

        # Create the state dictionary for the .pt file (excluding model args)
        save_dict = {
            'epoch': epoch + 1, # Save next epoch to start from
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            # DO NOT include 'model_args' here anymore
        }

        # Save the state dictionary
        try:
            torch.save(save_dict, checkpoint_path)
            # Optional: More controlled logging frequency
            log_freq = getattr(self.config, 'log_frequency', 100) # Default frequency
            if self.global_step % log_freq == 0 or getattr(self.config, 'debug_mode', False):
                save_msg = f"Checkpoint saved to {checkpoint_path}"
                save_msg += " (params.json updated)" if model_args_saved else " (params.json NOT updated)"
                logger.info(save_msg)
        except Exception as e:
            logger.error(f"Error saving checkpoint state to {checkpoint_path}: {e}", exc_info=True)


    def _load_checkpoint(self):
        """
        Loads the latest training state checkpoint (.pt file) from the save path.
        Assumes model architecture is already set up (using params.json loaded elsewhere).
        """
        # Assumes self.config.save_path is checked before calling _load_checkpoint in __init__
        save_path = self.config.save_path

        if not os.path.isdir(save_path):
             logger.error(f"Save directory {save_path} not found. Cannot load checkpoint.")
             return

        # Find checkpoint files
        checkpoints = [f for f in os.listdir(save_path) if f.startswith("checkpoint_") and f.endswith(".pt")]
        if not checkpoints:
            logger.info(f"No checkpoints (.pt files) found in {save_path}. Starting from scratch.")
            return

        # Sort checkpoints by epoch and step to find the latest
        def get_sort_key(filename):
            epoch_match = re.search(r"epoch_(\d+)", filename)
            step_match = re.search(r"step_(\d+)", filename)
            # Use large negative numbers for missing parts to sort them last if needed,
            # but typically checkpoints should have both epoch and step.
            epoch = int(epoch_match.group(1)) if epoch_match else -1
            step = int(step_match.group(1)) if step_match else -1
            return epoch, step

        try:
            checkpoints.sort(key=get_sort_key)
            latest_checkpoint_filename = checkpoints[-1]
            latest_checkpoint_path = os.path.join(save_path, latest_checkpoint_filename)
        except Exception as e:
             logger.error(f"Error sorting checkpoint files in {save_path}: {e}. Cannot determine latest checkpoint.", exc_info=True)
             return

        logger.info(f"Attempting to load checkpoint state from: {latest_checkpoint_path}")

        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=self.device)

            # Load training progress
            # If loading epoch 0, it means epoch 0 finished, so start at 1. Checkpoint saves epoch+1.
            self.start_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            logger.info(f"Checkpoint indicates resuming from epoch {self.start_epoch} at global step {self.global_step}")

            # Load model weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded model state dict.")
            else:
                 logger.error("Checkpoint missing 'model_state_dict'. Cannot load model weights.")
                 raise KeyError("Missing 'model_state_dict'") # Raise to prevent partial load

            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 logger.info("Loaded optimizer state dict.")
            else:
                 logger.warning("Checkpoint missing 'optimizer_state_dict'. Optimizer state not loaded.")

            # Load scheduler state
            if 'lr_scheduler_state_dict' in checkpoint:
                 self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                 logger.info("Loaded LR scheduler state dict.")
            else:
                 logger.warning("Checkpoint missing 'lr_scheduler_state_dict'. LR scheduler state not loaded.")

            # Load scaler state
            if self.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logger.info("Loaded GradScaler state dict.")
            elif self.use_amp:
                 logger.warning("AMP enabled, but no GradScaler state found or it was None in checkpoint.")

            # --- Model Args Check (Optional but recommended) ---
            # The TinyLlama.build method already loads params.json.
            # You could add a check here to ensure the loaded model's args match
            # the params.json in the checkpoint directory for extra safety,
            # but it's omitted here for simplicity as build handles the initial load.
            # Example check:
            # params_path = os.path.join(save_path, "params.json")
            # if os.path.exists(params_path):
            #     with open(params_path, 'r') as f:
            #         loaded_params = json.load(f)
            #     # Compare loaded_params with self.model.args (after converting self.model.args to a comparable dict)
            #     # Log warnings if they mismatch significantly.

            logger.info(f"Successfully loaded checkpoint and resumed state.")

        except FileNotFoundError:
             logger.error(f"Error: Checkpoint file not found at {latest_checkpoint_path}. It might have been deleted after listing.")
        except Exception as e:
             logger.error(f"Error loading checkpoint state from {latest_checkpoint_path}: {e}. Check file integrity or compatibility.", exc_info=True)
             # Resetting state might be necessary depending on the error
             logger.warning("Resetting training state (epoch, step) due to checkpoint load error.")
             self.start_epoch = 0
             self.global_step = 0


    def train(self, resume_from_checkpoint_override=False):
        """
        Executes the main training loop over the configured number of epochs.

        Args:
            resume_from_checkpoint_override (bool): If True, attempts to load a checkpoint
                even if config.resume_from_checkpoint was initially False. Requires
                config.save_path to be set. Defaults to False.
        """
        # Handle override for resuming
        if resume_from_checkpoint_override and not self.resume_enabled:
             if self.save_path_defined:
                 logger.info("Override: Attempting to resume training from checkpoint (config had resume=False).")
                 self._load_checkpoint() # Attempt to load if override is requested
             else:
                  logger.warning("Cannot resume (override): config.save_path not defined.")

        epochs_to_run = getattr(self.config, 'epochs', 1) # Default to 1 epoch if not specified
        logger.info(f"Starting training loop from epoch {self.start_epoch + 1} up to {epochs_to_run} epochs...")

        last_saved_loss = float('inf') # Track loss for filename

        for epoch in range(self.start_epoch, epochs_to_run):
            logger.info(f"\n===== Epoch {epoch + 1}/{epochs_to_run} =====")
            self.model.train() # Set model to training mode
            total_loss = 0.0
            epoch_start_time = time.time()
            # Use TQDM for progress visualization
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training", leave=False, unit="batch")

            for step, batch in enumerate(progress_bar):
                # Move batch data to the configured device
                try:
                    # Assuming batch is a dictionary with 'input_ids' and 'labels'
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                    targets = batch['labels'].to(self.device, non_blocking=True)
                except KeyError as e:
                     logger.error(f"Batch missing expected key: {e}. Check DataLoader output format.", exc_info=True)
                     continue # Skip batch if format is wrong
                except Exception as e:
                     logger.error(f"Error moving batch to device at step {step}: {e}", exc_info=True)
                     continue

                # Forward pass with Automatic Mixed Precision context
                with torch.autocast(device_type=self.device.type, dtype=self.pt_dtype, enabled=self.use_amp):
                    try:
                        # Assuming model takes TransformerInput or similar structure
                        # Adjust if your model takes input_ids directly
                        input_data = TransformerInput(tokens=input_ids, tokens_position=0) # Pos 0 for non-causal? Check model req.
                        outputs = self.model(input_data) # Removed use_cache=False, assume handled by model.train()
                        logits = outputs.logits # Expecting [batch, seq_len, vocab_size]

                        # Calculate loss using the pre-configured criterion
                        # Reshape logits and targets for CrossEntropyLoss:
                        # Logits: [batch * seq_len, vocab_size]
                        # Targets: [batch * seq_len]
                        loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                        # Check for invalid loss values
                        if loss is None or torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"Invalid loss ({loss}) encountered at global step {self.global_step}. Skipping backward pass for this batch.")
                            
                            # self.optimizer.zero_grad() # Optional: zero grad before skipping if needed
                            continue # Skip backward/optimizer step

                    except Exception as e:
                        logger.error(f"\nError during model forward or loss calculation at global step {self.global_step}. Error: {e}", exc_info=True)
                        print(f"\nError during model forward or loss calculation at global step {self.global_step}. Error: {e}", exc_info=True)
                        # Optional debugging prints:
                        # logger.info(f"Input IDs shape: {input_ids.shape}")
                        # logger.info(f"Targets shape: {targets.shape}")
                        # try: logger.info(f"Logits shape: {logits.shape}")
                        # except NameError: pass
                        continue # Skip batch on error

                # --- Debug Print Section (Optional) ---
                debug_print_freq = getattr(self.config, 'debug_print_freq', 10) # Print every N steps
                if getattr(self.config, 'debug_mode', False) and self.global_step % debug_print_freq == 0:
                    if logits is not None and self.tokenizer is not None and hasattr(self.criterion, 'ignore_index'):
                        try:
                            with torch.no_grad(): # Ensure no gradients calculated here
                                preds = torch.argmax(logits, dim=-1)[0] # Get predictions for first item in batch
                                max_len_print = 64 # Limit print length

                                input_to_print = input_ids[0, :max_len_print]
                                preds_to_print = preds[:max_len_print]
                                targets_to_print = targets[0, :max_len_print].clone() # Clone to avoid modifying original

                                ignore_idx = self.criterion.ignore_index
                                pad_token_id = self.tokenizer.pad_id if hasattr(self.tokenizer, 'pad_id') else ignore_idx # Fallback to ignore_idx if no pad_id

                                # Replace ignored target tokens with pad token for clearer decoding
                                targets_to_print[targets_to_print == ignore_idx] = pad_token_id

                                # Decode tokens to text
                                decoded_input = self.tokenizer.decode(input_to_print.tolist())
                                decoded_targets = self.tokenizer.decode(targets_to_print.tolist())
                                decoded_preds = self.tokenizer.decode(preds_to_print.tolist())

                                logger.info(f"\n--- Debug Step {self.global_step} (Batch Item 0 / First {max_len_print} Tokens) ---")
                                logger.info(f"Input : {decoded_input}")
                                logger.info(f"Target: {decoded_targets}")
                                logger.info(f"Pred  : {decoded_preds}")
                                logger.info(f"Loss  : {loss.item():.4f}" if loss is not None else "N/A")
                                logger.info(f"LR    : {self.lr_scheduler.get_last_lr()[0]:.2e}")
                                logger.info(f"--------------------------------------------------")

                        except AttributeError as e:
                            logger.warning(f"Debug print skipped: Missing attribute. Check tokenizer methods ('decode', 'pad_id') or criterion ('ignore_index'). Error: {e}")
                        except Exception as e:
                            logger.warning(f"Error during debug print at step {self.global_step}: {e}", exc_info=True)
                    else:
                         if self.global_step % debug_print_freq == 0: # Avoid logging every step if components missing
                             logger.warning(f"Debug print skipped at step {self.global_step}: Missing logits, tokenizer, or criterion.ignore_index.")
                # --- End Debug Print Section ---


                # --- Backward Pass and Optimization ---
                self.optimizer.zero_grad() # Zero gradients before backward pass
                # Scales loss. Calls backward() on scaled loss to create scaled gradients.
                self.scaler.scale(loss).backward()

                # Unscales gradients and applies clipping (if enabled)
                if self.max_grad_norm:
                    self.scaler.unscale_(self.optimizer) # Unscale gradients inplace
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called.
                # Otherwise, optimizer.step() is skipped.
                self.scaler.step(self.optimizer)

                # Updates the scale for next iteration.
                self.scaler.update()

                # Step the learning rate scheduler
                self.lr_scheduler.step()
                # --- End Optimization ---


                # --- Logging and Progress ---
                self.global_step += 1
                current_loss = loss.item()
                total_loss += current_loss
                last_saved_loss = current_loss # Update loss for potential checkpoint saving

                # Update progress bar
                progress_bar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{self.lr_scheduler.get_last_lr()[0]:.2e}", step=self.global_step)
                # --- End Logging ---

            # --- End of Epoch ---
            progress_bar.close()
            avg_train_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
            epoch_duration = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} | Time: {datetime.timedelta(seconds=int(epoch_duration))}")

            # --- Validation Step ---
            avg_val_loss_str = "N/A"
            if self.val_loader:
                val_loss = self.evaluate() # evaluate now uses self.criterion
                avg_val_loss_str = f"{val_loss:.4f}"
                logger.info(f"Epoch {epoch+1} Val Loss: {avg_val_loss_str}")
            else:
                logger.info("No validation loader provided, skipping validation.")
            # --- End Validation ---

            # --- Save Checkpoint ---
            # Consider saving based on validation loss improvement or fixed frequency
            save_freq = getattr(self.config, 'save_frequency_epochs', 1) # Save every epoch by default
            if (epoch + 1) % save_freq == 0:
                 self._save_checkpoint(epoch, last_saved_loss)
            # --- End Checkpoint ---

        logger.info("Training finished.")


    def evaluate(self):
        """
        Evaluates the model on the validation set using the pre-configured criterion.

        Returns:
            float: The average validation loss. Returns 0.0 if no validation loader exists.
        """
        if not self.val_loader:
            logger.info("Evaluation skipped: No validation data loader provided.")
            return 0.0

        self.model.eval() # Set model to evaluation mode
        total_val_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validation", leave=False, unit="batch")

        # Disable gradient calculations for evaluation
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch data to the configured device
                try:
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                    targets = batch['labels'].to(self.device, non_blocking=True)
                except KeyError as e:
                     logger.error(f"Validation batch missing expected key: {e}. Check DataLoader output format.", exc_info=True)
                     continue # Skip batch
                except Exception as e:
                     logger.error(f"Error moving validation batch to device: {e}", exc_info=True)
                     continue

                # Forward pass with AMP context (though gradients aren't computed)
                with torch.autocast(device_type=self.device.type, dtype=self.pt_dtype, enabled=self.use_amp):
                    try:
                        input_data = TransformerInput(tokens=input_ids, tokens_position=0)
                        outputs = self.model(input_data)
                        logits = outputs.logits

                        # Calculate loss using the pre-configured criterion
                        loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                        # Accumulate valid loss values
                        if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                             total_val_loss += loss.item()
                             progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                        else:
                             logger.warning(f"Invalid loss ({loss}) encountered during validation. Skipping batch contribution.")

                    except Exception as e:
                        logger.error(f"\nError during validation forward/loss calculation. Error: {e}", exc_info=True)
                        continue # Skip batch on error

        # Calculate average validation loss
        avg_val_loss = total_val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        return avg_val_loss