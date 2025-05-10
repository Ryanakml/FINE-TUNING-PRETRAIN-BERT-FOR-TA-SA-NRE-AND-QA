# BERT Model Implementation

This is a NumPy implementation of the BERT (Bidirectional Encoder Representations from Transformers) model that can be imported and used for pretraining and fine-tuning tasks.

## Overview

This implementation includes:

- Base BERT model architecture
- Masked Language Modeling (MLM) head for pretraining
- Next Sentence Prediction (NSP) head for pretraining
- Helper functions to easily create models with standard parameters

## Usage

### Importing the Model

```python
# Import the BERT model and helper functions
from Bert_Model import create_bert_model, BertModel, BertForPretraining
```

### Creating a Model for Pretraining

```python
# Create a BERT model with pretraining heads (MLM and NSP)
model = create_bert_model(
    vocab_size=30000,
    max_seq_length=512,
    d_model=768,       # Hidden size
    num_heads=12,      # Number of attention heads
    d_ff=3072,         # Feed-forward layer size
    num_layers=12,     # Number of transformer layers
    for_pretraining=True
)

# Forward pass with input token IDs
outputs = model.forward(token_ids)

# Access different outputs
mlm_logits = outputs['mlm_logits']         # For masked token prediction
nsp_logits = outputs['nsp_logits']         # For next sentence prediction
sequence_output = outputs['sequence_output'] # Full sequence representations
pooled_output = outputs['pooled_output']    # [CLS] token representation
```

### Creating a Base Model (without pretraining heads)

```python
# Create just the base BERT model without MLM and NSP heads
base_model = create_bert_model(
    vocab_size=30000,
    max_seq_length=512,
    d_model=768,
    num_heads=12,
    d_ff=3072,
    num_layers=12,
    for_pretraining=False
)

# Forward pass returns sequence output directly
sequence_output = base_model.forward(token_ids)
```

## Example

See `example_usage.py` for a complete example of how to use the model.

## Model Architecture

- **WordEmbedding**: Token embedding lookup table
- **PositionalEncoding**: Sinusoidal position embeddings
- **MultiheadAttention**: Multi-head self-attention mechanism
- **FeedForward**: Position-wise feed-forward network with GELU activation
- **BertBlock**: Transformer encoder block with residual connections and layer normalization
- **BertModel**: Base BERT model that stacks multiple encoder blocks
- **MaskedLMHead**: Prediction head for the masked language modeling task
- **NSPHead**: Prediction head for the next sentence prediction task
- **BertForPretraining**: Combines the base model with MLM and NSP heads

## Notes

- This implementation uses NumPy for all operations
- For pretraining, you'll need to implement masking logic and loss functions
- The model is designed to be compatible with standard BERT pretraining objectives