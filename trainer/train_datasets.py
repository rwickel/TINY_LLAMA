import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PyTorch Dataset Classes ---
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long) if not isinstance(input_ids, torch.Tensor) else input_ids
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long) if not isinstance(attention_mask, torch.Tensor) else attention_mask
        self.labels = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

class TinyStoriesDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings['input_ids'].size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['labels'][idx]
        }

# --- Helper function for padding/truncation ---
def pad_and_truncate(token_ids: List[int], max_length: int, pad_id: int) -> (List[int], List[int]):
    """Pads or truncates a list of token IDs and creates attention mask."""
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
        attention_mask = [1] * max_length
    else:
        attention_mask = [1] * len(token_ids) + [0] * (max_length - len(token_ids))
        token_ids = token_ids + [pad_id] * (max_length - len(token_ids))
    return token_ids, attention_mask

# --- Adapted default_dataset function ---
def default_dataset(tokenizer, max_seq_length: int):
    """Creates a small default dataset for unsupervised pre-training."""
    logger.info("Creating default dataset...")

    try:
        bos_string = tokenizer.decode([tokenizer.bos_id])
        eos_string = tokenizer.decode([tokenizer.eos_id])
        pad_id = tokenizer.pad_id
        if pad_id is None or pad_id < 0:
             raise ValueError("Tokenizer must have a valid pad_id set.")
    except Exception as e:
        logger.error(f"Error getting BOS/EOS/PAD from tokenizer: {e}")
        raise

    logger.info(f"Using BOS: '{bos_string}' ({tokenizer.bos_id}), EOS: '{eos_string}' ({tokenizer.eos_id}), PAD: {pad_id}")

    texts = [
        "The rabbit hopped through the green field.",
        "The rabbit hopped over the trunk.",
        "The cat slept on the warm couch.",
        "The dog chased the red ball.",
        "The bird sang on the tree.",
        "Tom rode his bike to school.",
        "The fish swam in the pond.",
        "Emma ate a sweet cupcake.",
        "The butterfly flew over the flowers.",
        "The teddy bear sat on the bed.",
    ]

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    logger.info(f"Processing {len(texts)} text samples...")
    for text in tqdm(texts, desc="Tokenizing default dataset"):
        formatted_text = f"{bos_string}{text}{eos_string}"

        token_ids = tokenizer.encode(
            formatted_text,
            bos=True,
            eos=True,
            allowed_special="all"
        )

        padded_token_ids, attention_mask = pad_and_truncate(token_ids, max_seq_length, pad_id)

        # Create labels with padding tokens (no -100)
        labels = padded_token_ids[1:] + [pad_id]
        
        # Mask labels where input was padding
        for i in range(max_seq_length):
            if attention_mask[i] == 0:
                labels[i] = pad_id

        assert len(labels) == max_seq_length

        all_input_ids.append(padded_token_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    logger.info("Stacking tensors...")
    all_input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_masks_tensor = torch.tensor(all_attention_masks, dtype=torch.long)
    all_labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    logger.info("Default dataset creation complete.")
    return TrainDataset(all_input_ids_tensor, all_attention_masks_tensor, all_labels_tensor)

# --- Adapted load_tinystories_dataset function ---
def load_tinystories_dataset(tokenizer, max_seq_length: int, split="train", max_samples=None, show_progress=True, skip_first=0):
    """Loads and processes the TinyStories dataset."""
    logger.info(f"Loading TinyStories dataset (split={split}, max_samples={max_samples}, skip={skip_first})...")

    try:
        bos_string = tokenizer.decode([tokenizer.bos_id])
        eos_string = tokenizer.decode([tokenizer.eos_id])
        pad_id = tokenizer.pad_id
        if pad_id is None or pad_id < 0:
             raise ValueError("Tokenizer must have a valid pad_id set.")
    except Exception as e:
        logger.error(f"Error getting BOS/EOS/PAD from tokenizer: {e}")
        raise

    logger.info(f"Using BOS: '{bos_string}' ({tokenizer.bos_id}), EOS: '{eos_string}' ({tokenizer.eos_id}), PAD: {pad_id}")

    use_streaming = max_samples is None or max_samples > 50000
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=use_streaming)

    if skip_first > 0:
        logger.info(f"Skipping first {skip_first} samples...")
        if use_streaming:
            dataset = dataset.skip(skip_first)
        else:
            skip_first = min(skip_first, len(dataset))
            dataset = dataset.select(range(skip_first, len(dataset)))

    if max_samples is not None:
        logger.info(f"Taking max {max_samples} samples...")
        if use_streaming:
            dataset = dataset.take(max_samples)
        else:
            effective_max = min(max_samples, len(dataset))
            dataset = dataset.select(range(effective_max))

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    if show_progress:
         try:
             total = len(dataset) if not use_streaming else max_samples
             pbar = tqdm(total=total, desc=f"Processing TinyStories ({split})")
         except TypeError:
             pbar = tqdm(desc=f"Processing TinyStories ({split})")
    else:
         pbar = None

    processed_count = 0
    for example in dataset:
        text = example.get('text')
        if not text:
            logger.warning("No text found in example, skipping...")
            if pbar: pbar.update(1)
            continue

        try:
            token_ids = tokenizer.encode(
                text,
                bos=True,
                eos=True,
                allowed_special="all"
            )
        except Exception as e:
            logger.warning(f"Tokenization failed for an example, skipping. Error: {e}")
            if pbar: pbar.update(1)
            continue

        padded_token_ids, attention_mask = pad_and_truncate(token_ids, max_seq_length, pad_id)

        # Create labels with padding tokens (no -100)
        labels = padded_token_ids[1:] + [pad_id]
        
        # Mask labels where input was padding
        for i in range(max_seq_length):
            if attention_mask[i] == 0:
                labels[i] = pad_id

        assert len(labels) == max_seq_length

        all_input_ids.append(padded_token_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

        processed_count += 1
        if pbar: pbar.update(1)
        if max_samples is not None and use_streaming and processed_count >= max_samples:
            break

    if pbar: pbar.close()
    logger.info(f"Processed {processed_count} samples.")

    if not all_input_ids:
         logger.error("No data processed, returning empty dataset!")
         return TrainDataset(torch.empty((0, max_seq_length), dtype=torch.long),
                          torch.empty((0, max_seq_length), dtype=torch.long),
                          torch.empty((0, max_seq_length), dtype=torch.long))

    logger.info("Stacking tensors for TinyStories dataset...")
    all_input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_masks_tensor = torch.tensor(all_attention_masks, dtype=torch.long)
    all_labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    encodings = {
        'input_ids': all_input_ids_tensor,
        'attention_mask': all_attention_masks_tensor,
        'labels': all_labels_tensor
    }

    logger.info("TinyStories dataset creation complete.")
    return TinyStoriesDataset(encodings)