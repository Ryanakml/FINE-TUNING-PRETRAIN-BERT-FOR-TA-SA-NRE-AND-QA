# Example of how to import and use the BERT model for pretraining

import numpy as np
from Bert_Model import create_bert_model, BertModel, BertForPretraining

# Example of creating a BERT model for pretraining
def example_pretraining():
    # Create a small BERT model for demonstration
    vocab_size = 30000
    max_seq_length = 512
    d_model = 64      # Smaller dimension for testing
    num_heads = 4     # Must be a divisor of d_model
    d_ff = 128
    num_layers = 2    # Fewer layers for faster execution
    
    # Create model instance using the helper function
    model = create_bert_model(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        for_pretraining=True
    )
    
    # Example input (batch_size=2, seq_len=6)
    # In real pretraining, you would have masked tokens and NSP labels
    token_ids = np.array([
        [101, 2054, 2003, 1996, 2307, 102],  # [CLS] this is the test [SEP]
        [101, 2023, 2003, 1037, 8331, 102]   # [CLS] it is a example [SEP]
    ])
    
    # Forward pass
    outputs = model.forward(token_ids)
    
    # Access the different outputs
    mlm_logits = outputs['mlm_logits']
    nsp_logits = outputs['nsp_logits']
    sequence_output = outputs['sequence_output']
    pooled_output = outputs['pooled_output']
    
    print(f"MLM logits shape: {mlm_logits.shape}")       # Should be (2, 6, 30000)
    print(f"NSP logits shape: {nsp_logits.shape}")       # Should be (2, 2)
    print(f"Sequence output shape: {sequence_output.shape}")  # Should be (2, 6, 64)
    print(f"Pooled output shape: {pooled_output.shape}")     # Should be (2, 64)
    
    return model

# Example of creating a base BERT model (without pretraining heads)
def example_base_model():
    # Create a base BERT model (without MLM and NSP heads)
    model = create_bert_model(
        vocab_size=30000,
        max_seq_length=512,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        for_pretraining=False  # Just get the base model
    )
    
    # Example input
    token_ids = np.array([
        [101, 2054, 2003, 1996, 2307, 102],
        [101, 2023, 2003, 1037, 8331, 102]
    ])
    
    # Forward pass
    sequence_output = model.forward(token_ids)
    
    print(f"Base model output shape: {sequence_output.shape}")  # Should be (2, 6, 64)
    
    return model

if __name__ == "__main__":
    print("Testing BERT model for pretraining...")
    pretraining_model = example_pretraining()
    
    print("\nTesting base BERT model...")
    base_model = example_base_model()
    
    print("\nBERT models successfully created and tested!")