# trainer/squad_data_utils.py

import os
import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset, DataLoader
# Import the generic tokenizer and config class from other modules
from .config import TrainingConfig # Assuming TrainingConfig might be needed, adjust if not


from typing import Optional, List, Dict, Any

# --- SQuAD Dataset Class ---
class SQuADDataset(Dataset):
    """
    A PyTorch Dataset for loading and preparing the SQuAD dataset for
    Causal Language Modeling.

    Args:
        tokenizer: The tokenizer to use for encoding the text.
        config: TrainingConfig object (or similar object) containing training parameters.
        split: The dataset split to load ("train" or "validation").
        max_samples: Maximum number of samples to load from the split (optional).
    """

    def __init__(self, tokenizer, config, split: str = "train", max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.max_length = config.max_seq_length  # Assuming max_seq_length is in config
        self.data = self._load_and_prepare_squad(max_samples)

    def __len__(self):
        return len(self.data) if self.data else 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing 'input_ids', 'attention_mask', and 'labels'
        for the given index.
        """
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)  # Or adjust based on your task
        }

    def _prepare_squad(self, examples: Dict[str, List[Any]]):
        """
        Internal helper to format SQuAD examples into 'Context: C Question: Q Answer: A<EOS>'
        for Causal Language Modeling. Creates 'text_to_tokenize' column.
        Takes the *first* answer provided.
        """
        texts = []

        contexts = examples.get('context', [])
        questions = examples.get('question', [])
        answers_list = examples.get('answers', [])

        # Basic validation
        if not (len(contexts) == len(questions) == len(answers_list)):
            print(
                "Warning: Mismatch in lengths of context/question/answers in SQuAD batch. Skipping batch.")
            # Return structure expected by .map even if empty
            return {"text_to_tokenize": [""] * len(next(iter(examples.values()), []))}

        for context, question, answers in zip(contexts, questions, answers_list):
            # SQuAD answers is a dict {'text': [...], 'answer_start': [...]}
            # We'll take the first answer text if available
            answer_text = ""
            if isinstance(answers, dict) and 'text' in answers and answers['text']:
                answer_text = answers['text'][0]
            elif isinstance(answers, list) and answers:  # Handle if it's somehow just a list of strings
                answer_text = answers[0]

            # Ensure all parts are strings
            context_str = str(context) if context is not None else ""
            question_str = str(question) if question is not None else ""
            answer_str = str(answer_text) if answer_text is not None else ""

            formatted_text = (
                
                f"<CTX> {context_str} </CTX>"
                f"<Q> {question_str} </Q>"
                f"<A> {answer_str} </A>"
                
            )
            texts.append(formatted_text)

        return {"text_to_tokenize": texts}

    
    def generic_tokenize_fn(self, examples, tokenizer, config, ignore_index=-100):
        """
        Tokenizes examples and creates labels for causal LM training,
        where labels start from <A> token onward.
        """
        if "text_to_tokenize" not in examples or not examples["text_to_tokenize"]:
            return {'input_ids': [], 'attention_mask': [], 'labels': []}

        texts = [str(t) for t in examples["text_to_tokenize"] if t is not None]

        tokenized = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=config.max_seq_length,
            add_special_tokens=True,
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        targets = input_ids.clone()

        targets[:-1] = input_ids[1:]
        targets[-1] = tokenizer.pad_token_id  # Last token doesn't predict anything

        # Ignore pad tokens in the loss
        targets[input_ids == tokenizer.pad_token_id] = -100      

        # # Get <A> token ID
        # start_a_token_id = tokenizer.convert_tokens_to_ids("<A>")
        # pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        # # Set ignore_index for everything BEFORE first <A>
        # for i in range(input_ids.size(0)):
        #     a_token_pos = (input_ids[i] == start_a_token_id).nonzero(as_tuple=True)
        #     if len(a_token_pos[0]) > 0:
        #         start = a_token_pos[0].item()
        #         labels[i, :start] = ignore_index
        #     else:
        #         # If no <A> token found, ignore the whole sequence
        #         labels[i] = ignore_index

        #     # Also ignore padding positions
        #     labels[i][input_ids[i] == pad_token_id] = ignore_index

        tokenized["labels"] = targets
        return tokenized

    
    
    def _load_and_prepare_squad(self, max_samples: Optional[int] = None) -> List[Dict[str, List[int]]]:
        """
        Loads the SQuAD dataset (potentially limited by max_samples parameter),
        prepares it for Causal LM training, and tokenizes using generic_tokenize_fn.

        Returns:
            List[Dict[str, List[int]]]: A list of dictionaries, where each dictionary
            contains 'input_ids', 'attention_mask', and 'labels' (lists of ints).
            Returns an empty list if loading/processing fails.
        """

        print("Executing SQuAD data preparation within SQuADDataset class.")

        # --- Determine the split string based on the max_samples parameter ---
        if max_samples and max_samples > 0:
            split_string = f"{self.split}[:{max_samples}]"  # Use self.split
            print(f"Loading limited SQuAD data: {split_string}")
        else:
            split_string = self.split  # Use self.split
            print(f"Loading full SQuAD {self.split} split.")

        print("Loading SQuAD dataset...")
        try:
            # Use the determined split_string here
            raw_dataset = load_dataset("squad", split=split_string, trust_remote_code=True)
            print(f"Loaded SQuAD data with {len(raw_dataset)} samples.")
            if len(raw_dataset) == 0:
                print("Fatal: SQuAD data loaded empty (check split string or limit).")
                return []  # Return empty list
        except Exception as e:
            print(f"Fatal: Failed to load SQuAD dataset. Error: {e}")
            return []  # Return empty list

        # Prepare the dataset
        print("Preparing SQuAD data for CLM format...")
        prepare_fn_with_tokenizer = lambda exs: self._prepare_squad(exs)
        original_columns = raw_dataset.column_names
        num_proc = os.cpu_count()
        prepared_dataset = raw_dataset.map(
            prepare_fn_with_tokenizer,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
            desc="Preparing SQuAD data"
        )

        # Validate preparation step
        if "text_to_tokenize" not in prepared_dataset.column_names or len(prepared_dataset) == 0:
            print("Fatal: SQuAD preparation failed...")
            return []  # Return empty list

        # Tokenize the prepared data
        print("Tokenizing prepared data using generic_tokenize_fn...")
        if not callable(self.generic_tokenize_fn):
            print("Fatal: generic_tokenize_fn imported from data_utils is not callable.")
            return []  # Return empty list
        tokenize_fn_with_args = lambda exs: self.generic_tokenize_fn(exs, self.tokenizer, self.config)

        try:
            tokenized_data = prepared_dataset.map(
                tokenize_fn_with_args, batched=True, num_proc=num_proc,
                remove_columns=["text_to_tokenize"], desc="Tokenizing SQuAD data"
            )

            if not tokenized_data or 'input_ids' not in tokenized_data.column_names:
                print("Fatal: Tokenization failed for data or 'input_ids' missing.")
                return []  # Return empty list

            print(f"Tokenized data with {len(tokenized_data)} samples.")
            print(f"Tokenized features: {tokenized_data.features}")

            # Convert to a list of dictionaries for easier batching
            return [{
                'input_ids': example['input_ids'],
                'attention_mask': example['attention_mask'],
                'labels': example['labels']
            } for example in tokenized_data]

        except Exception as e:
            print(f"Fatal: Failed during tokenization map operation. Error: {e}")
            return []  # Return empty list