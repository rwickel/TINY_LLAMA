# Advanced Transformer Model Implementation

This repository contains a PyTorch implementation of an advanced Transformer-based language model (`LLM`). It incorporates several modern techniques and features, including Mixture of Experts (MoE), Rotary Position Embeddings (RoPE), optional vision integration, and more. The architecture is defined primarily in `model.py` using core layer implementations from `layers.py`.

## Overview

This implementation provides a flexible and configurable foundation for building large language models. It follows the standard Transformer architecture but replaces or enhances components with recent advancements for potentially improved performance, scalability, and multimodality.

**Core Files:**

* `model.py`: Defines the main `LLM` class and the `TransformerBlock` class representing a single layer. Handles model assembly, embedding, normalization, and output projection.
* `layers.py`: Provides the implementations for essential layers like `Attention`, `FeedForward` (SwiGLU), `RMSNorm`, and RoPE helper functions.
* `model/args.py`: (Assumed) Defines the `ModelArgs` dataclass used for configuring the model hyperparameters.
* `model/moe.py`: (Assumed) Contains the `MoE` layer implementation.
* `vision/embedding.py`: (Assumed) Contains the `VisionEmbeddings` implementation for multimodal capabilities.

## Key Features

* **Transformer Architecture:** Based on the standard Transformer (`LLM`, `TransformerBlock`) with pre-normalization.
* **Rotary Position Embeddings (RoPE):** Implements RoPE for incorporating positional information within the attention mechanism. Includes frequency scaling options (`apply_scaling`) for potential context length extension.
* **Mixture of Experts (MoE):** Optionally replaces standard FeedForward layers with MoE layers (`model.moe.MoE`) for increased model capacity with potentially similar computational cost during inference. Configurable per layer.
* **Advanced Attention:**
    * Supports Multi-Head Attention (MHA) and Grouped-Query Attention (GQA).
    * Includes efficient KV Caching for autoregressive generation.
    * Optional Query-Key Normalization (`use_qk_norm`).
    * Utilizes `F.scaled_dot_product_attention` for optimized computation.
* **Feed-Forward Network:** Implements the SwiGLU (Swish Gated Linear Unit) variant (`FeedForward` class).
* **RMS Normalization:** Uses `RMSNorm` instead of LayerNorm for normalization.
* **Vision Integration:** Supports optional image input via `VisionEmbeddings`, allowing for multimodal text-image processing.
* **NoPE Layers:** Allows specific layers to operate without explicit positional embeddings (`args.nope_layer_interval`).
* **Chunked Local Attention:** Includes helper functions (`create_chunked_attention_mask`) to support local attention patterns, potentially useful with specific RoPE variants (like iRoPE).
* **Configuration:** Highly configurable via the `ModelArgs` class.
* **Checkpoint Compatibility:** Includes `load_hook` methods in `Attention`, `FeedForward`, `TransformerBlock`, and `LLM` to handle loading weights from checkpoints with potentially different naming conventions or structures.
* **Model Parallelism:** Includes stubs and potential dependencies (`fairscale`) suggesting support for model parallelism techniques (though specifics like `VocabParallelEmbedding` might be commented out).

## Architecture Details

### `LLM` Class (`model.py`)

* The main container class for the entire language model.
* Initializes token embeddings (`tok_embeddings`) and optionally vision embeddings/projection (`vision_embeddings`, `vision_projection`).
* Stacks multiple `TransformerBlock` layers (`self.layers`).
* Applies final normalization (`self.norm`).
* Projects final hidden states to vocabulary logits (`self.output`).
* Precomputes RoPE frequencies (`self.freqs_cis`).
* Handles merging of text and image embeddings based on input masks.
* Creates attention masks (global causal and optional local chunked).

### `TransformerBlock` Class (`model.py`)

* Represents a single layer within the `LLM`.
* Follows a pre-normalization structure:
    1.  `RMSNorm` (`attention_norm`)
    2.  `Attention` (MHA/GQA with RoPE/QKNorm/NoPE logic based on layer ID and config)
    3.  Residual Connection
    4.  `RMSNorm` (`ffn_norm`)
    5.  `FeedForward` or `MoE` (based on layer ID and config)
    6.  Residual Connection
* Selects appropriate attention mask (global or local) based on configuration.

### Core Layers (`layers.py`)

* **`RMSNorm`**: Efficient normalization layer with a learnable scale parameter.
* **RoPE Functions**: Utilities for precomputing, scaling, and applying rotary embeddings to queries and keys.
* **`Attention`**: Implements MHA/GQA, handling RoPE application, optional QK Norm, KV Caching via non-persistent buffers, and GQA head repetition.
* **`FeedForward`**: Implements the SwiGLU FFN.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install Dependencies:** Ensure you have PyTorch installed. Other dependencies might include `fairscale` (check requirements).
    ```bash
    # Example: Create a virtual environment
    # python -m venv venv
    # source venv/bin/activate
    pip install torch # Add other specific dependencies like fairscale if needed
    # Potentially: pip install -r requirements.txt (if provided)
    ```

## Usage

1.  **Configuration:** Define your model configuration using the `ModelArgs` class (likely defined in `model/args.py`). This controls dimensions, number of layers, heads, MoE parameters, RoPE settings, vision args, etc.
2.  **Instantiation:** Create an instance of the `LLM` class, passing the configured `ModelArgs`.
    ```python
    from model.args import ModelArgs # Assuming path
    from model import LLM
    from model.datatypes import TransformerInput # Assuming path

    # 1. Load or define ModelArgs
    model_args = ModelArgs(...) # Fill with desired configuration

    # 2. Instantiate the model
    model = LLM(model_args)

    # 3. Load pre-trained weights (example)
    # checkpoint = torch.load("path/to/checkpoint.pt", map_location="cpu")
    # model.load_state_dict(checkpoint, strict=False) # Hooks handle compatibility
    # model.eval() # Set to evaluation mode if using for inference

    # 4. Prepare input (example)
    # input_tokens = torch.tensor([[...]], dtype=torch.long)
    # input_pos = torch.arange(0, input_tokens.shape[1])
    # transformer_input = TransformerInput(tokens=input_tokens, tokens_position=input_pos)

    # 5. Forward pass
    # with torch.inference_mode(): # Or model.forward(..., inference=True)
    #     output = model(transformer_input)
    #     logits = output.logits
    ```
3.  **Training/Inference:** Integrate the model into your training or inference pipeline. Remember to handle KV caching correctly during autoregressive generation.

