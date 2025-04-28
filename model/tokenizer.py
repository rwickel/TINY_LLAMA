# model/tokenizer.py

import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    cast,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe

logger = getLogger(__name__)


# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000



def get_reserved_special_tokens(name, count, start_index=0):
    return [f"<|{name}_reserved_special_token_{i}|>" for i in range(start_index, start_index + count)]


# 200005, ..., 200079
TEXT_POST_TRAIN_SPECIAL_TOKENS = [
    "<|header_start|>",
    "<|header_end|>",
    "<|eom|>",
    "<|eot|>",
    "<|step|>",
    "<|text_post_train_reserved_special_token_0|>",
    "<|text_post_train_reserved_special_token_1|>",
    "<|text_post_train_reserved_special_token_2|>",
    "<|text_post_train_reserved_special_token_3|>",
    "<|text_post_train_reserved_special_token_4|>",
    "<|text_post_train_reserved_special_token_5|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|finetune_right_pad|>",
] + get_reserved_special_tokens(
    "text_post_train", 61, 8
)  # <|text_post_train_reserved_special_token_8|>, ..., <|text_post_train_reserved_special_token_68|>

# 200080, ..., 201133
VISION_SPECIAL_TOKENS = [
    "<|image_start|>",
    "<|image_end|>",
    "<|vision_reserved_special_token_0|>",
    "<|vision_reserved_special_token_1|>",
    "<|tile_x_separator|>",
    "<|tile_y_separator|>",
    "<|vision_reserved_special_token_2|>",
    "<|vision_reserved_special_token_3|>",
    "<|vision_reserved_special_token_4|>",
    "<|vision_reserved_special_token_5|>",
    "<|image|>",
    "<|vision_reserved_special_token_6|>",
    "<|patch|>",
] + get_reserved_special_tokens(
    "vision", 1041, 7
)  # <|vision_reserved_special_token_7|>, ..., <|vision_reserved_special_token_1047|>

# 201134, ..., 201143
REASONING_SPECIAL_TOKENS = [
    "<|reasoning_reserved_special_token_0|>",
    "<|reasoning_reserved_special_token_1|>",
    "<|reasoning_reserved_special_token_2|>",
    "<|reasoning_reserved_special_token_3|>",
    "<|reasoning_reserved_special_token_4|>",
    "<|reasoning_reserved_special_token_5|>",
    "<|reasoning_reserved_special_token_6|>",
    "<|reasoning_reserved_special_token_7|>",
    "<|reasoning_thinking_start|>",
    "<|reasoning_thinking_end|>",
]

CUSTOM_SPECIAL_TOKENS = (
    TEXT_POST_TRAIN_SPECIAL_TOKENS + VISION_SPECIAL_TOKENS + REASONING_SPECIAL_TOKENS
)

BASIC_SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
]


class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 2048

        
    def __init__(self, base_model_name: str = "cl100k_base"):
        """
        Initializes the Tokenizer with a specified Tiktoken base model
        and custom special tokens.

        Args:
            base_model_name (str): The name of the Tiktoken base encoding to use
                                   (e.g., "cl100k_base", "o200k_base").
        """
        # 1. Get the base encoding
        try:
            base_enc = tiktoken.get_encoding(base_model_name)
        except Exception as e:
            logger.error(f"Failed to get base encoding '{base_model_name}': {e}")
            raise

        # 2. Determine num_base_tokens from the base encoding
        num_base_tokens = base_enc.n_vocab
        logger.info(f"Using base model '{base_model_name}' with {num_base_tokens} base tokens.")

        # 3. Define the full list of custom special tokens
        explicit_special_tokens = BASIC_SPECIAL_TOKENS + CUSTOM_SPECIAL_TOKENS
        assert len(set(explicit_special_tokens)) == len(explicit_special_tokens), "Duplicate special tokens found"
        assert len(explicit_special_tokens) <= self.num_reserved_special_tokens, "More explicit special tokens than reserved space"

        # Add reserved placeholder tokens
        num_placeholders = self.num_reserved_special_tokens - len(explicit_special_tokens)
        reserved_tokens = [
            f"<|reserved_special_token_{i}|>" for i in range(num_placeholders)
        ]
        full_special_list = explicit_special_tokens + reserved_tokens

        # 4. Create the special token dictionary with IDs starting after base tokens
        # *** Corrected part ***
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(full_special_list)
        }

        # 5. Create the final Encoding object
        # Use a descriptive name for the combined encoding
        custom_name = f"{base_model_name}_custom"
        try:
            self.model = tiktoken.Encoding(
                name=custom_name,
                # Use pattern and merges from the base encoding
                pat_str=base_enc._pat_str,
                mergeable_ranks=base_enc._mergeable_ranks,
                # Add the custom special tokens
                special_tokens=self.special_tokens
            )
        except Exception as e:
             logger.error(f"Failed to create custom tiktoken.Encoding '{custom_name}': {e}")
             raise

        # 6. Set vocabulary size and special token IDs from the final model
        self.n_words: int = self.model.n_vocab # Total size including base and special
        logger.info(f"Total vocabulary size (incl. special tokens): {self.n_words}")

        # --- Assign specific token IDs ---
        # Use .get() for safety in case a token isn't in the final map
        self.bos_id: int = self.special_tokens.get("<|begin_of_text|>")
        self.eos_id: int = self.special_tokens.get("<|end_of_text|>")
        self.pad_id: int = self.special_tokens.get("<|finetune_right_pad|>")
        self.eot_id: int = self.special_tokens.get("<|eot|>")
        self.eom_id: int = self.special_tokens.get("<|eom|>")
        self.thinking_start_id: int = self.special_tokens.get("<|reasoning_thinking_start|>")
        self.thinking_end_id: int = self.special_tokens.get("<|reasoning_thinking_end|>")

        # Check if any essential tokens were missing
        missing_tokens = [name for name, val in [("bos", self.bos_id), ("eos", self.eos_id), ("pad", self.pad_id), ("eot", self.eot_id), ("eom", self.eom_id)] if val is None]
        if missing_tokens:
            raise ValueError(f"Essential special tokens missing from final map: {missing_tokens}")               
        
        self.stop_tokens = [
            self.eos_id,
            self.eom_id,
            self.eot_id,
        ]        

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_special ("all"|set[str]): allowed special tokens in string
            disallowed_special ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        if allowed_special is None:
            allowed_special = set()
        assert type(s) is str

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

# --- Example Usage ---
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize with cl100k_base
    print("\n--- Initializing with cl100k_base ---")
    tokenizer_cl100k = Tokenizer(base_model_name="cl100k_base")   

    text = "Hello world!"
    print(f"Original text: {text}")

    # Encode allowing special tokens
    encoded_special = tokenizer_cl100k.encode(text, bos=True, eos=True, allowed_special="all")
    print(f"Encoded (BOS+EOS, special allowed): {encoded_special}")
    decoded_special = tokenizer_cl100k.decode(encoded_special)
    print(f"Decoded: {decoded_special}")      
    