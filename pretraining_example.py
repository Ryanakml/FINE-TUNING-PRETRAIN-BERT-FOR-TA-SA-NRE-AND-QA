# Example of how to prepare data and use the BERT model for pretraining

import numpy as np
from Bert_Model import create_bert_model

# Helper function to create masked input for MLM task
def create_mlm_data(token_ids, mask_prob=0.15, vocab_size=30000):
    """
    Create masked input for the Masked Language Modeling task.
    
    Args:
        token_ids: Original token IDs [batch_size, seq_len]
        mask_prob: Probability of masking a token (default: 0.15)
        vocab_size: Size of vocabulary
        
    Returns:
        masked_input: Input with masked tokens
        mlm_labels: Labels for masked positions (-100 for non-masked positions)
    """
    masked_input = token_ids.copy()
    mlm_labels = np.full_like(token_ids, -100)  # -100 is ignored in loss calculation
    
    for i in range(token_ids.shape[0]):
        # Don't mask special tokens (assuming 0=PAD, 101=CLS, 102=SEP)
        special_tokens = [0, 101, 102]
        mask_candidates = [j for j, token in enumerate(token_ids[i]) if token not in special_tokens]
        
        # Randomly select tokens to mask
        n_to_mask = max(1, int(len(mask_candidates) * mask_prob))
        to_mask = np.random.choice(mask_candidates, size=n_to_mask, replace=False)
        
        for pos in to_mask:
            # Save original token for the label
            mlm_labels[i, pos] = token_ids[i, pos]
            
            # 80% of the time, replace with [MASK] token (assuming 103 is [MASK])
            # 10% of the time, replace with random token
            # 10% of the time, keep the original token
            prob = np.random.random()
            if prob < 0.8:
                masked_input[i, pos] = 103  # [MASK] token
            elif prob < 0.9:
                masked_input[i, pos] = np.random.randint(1, vocab_size)  # Random token
            # else: keep the original token
    
    return masked_input, mlm_labels

# Helper function to create data for NSP task
def create_nsp_data(sentences_a, sentences_b, is_next_labels):
    """
    Create input for Next Sentence Prediction task.
    
    Args:
        sentences_a: First sentences [batch_size, seq_len_a]
        sentences_b: Second sentences [batch_size, seq_len_b]
        is_next_labels: Binary labels indicating if sentence B follows sentence A
        
    Returns:
        token_ids: Combined token IDs with [CLS] and [SEP] tokens
        segment_ids: Segment IDs (0 for sentence A, 1 for sentence B)
        nsp_labels: NSP labels
    """
    batch_size = len(sentences_a)
    max_len_a = max(len(s) for s in sentences_a)
    max_len_b = max(len(s) for s in sentences_b)
    
    # Total sequence length: [CLS] + sentence A + [SEP] + sentence B + [SEP]
    total_len = 1 + max_len_a + 1 + max_len_b + 1
    
    token_ids = np.zeros((batch_size, total_len), dtype=np.int32)
    segment_ids = np.zeros((batch_size, total_len), dtype=np.int32)
    attention_mask = np.zeros((batch_size, total_len), dtype=np.int32)
    
    for i in range(batch_size):
        # [CLS] token
        token_ids[i, 0] = 101  # [CLS] token
        attention_mask[i, 0] = 1
        
        # Sentence A
        len_a = len(sentences_a[i])
        token_ids[i, 1:1+len_a] = sentences_a[i]
        attention_mask[i, 1:1+len_a] = 1
        
        # [SEP] token after sentence A
        token_ids[i, 1+len_a] = 102  # [SEP] token
        attention_mask[i, 1+len_a] = 1
        
        # Sentence B
        len_b = len(sentences_b[i])
        token_ids[i, 2+len_a:2+len_a+len_b] = sentences_b[i]
        segment_ids[i, 2+len_a:2+len_a+len_b] = 1  # Segment 1 for sentence B
        attention_mask[i, 2+len_a:2+len_a+len_b] = 1
        
        # [SEP] token after sentence B
        token_ids[i, 2+len_a+len_b] = 102  # [SEP] token
        segment_ids[i, 2+len_a+len_b] = 1  # Segment 1 for [SEP] after sentence B
        attention_mask[i, 2+len_a+len_b] = 1
    
    return token_ids, segment_ids, attention_mask, is_next_labels

# Example of pretraining loop
def pretraining_example():
    # Create a small BERT model for demonstration
    vocab_size = 30000
    max_seq_length = 512
    d_model = 64      # Smaller dimension for testing
    num_heads = 4     # Must be a divisor of d_model
    d_ff = 128
    num_layers = 2    # Fewer layers for faster execution
    
    # Create model instance
    model = create_bert_model(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        for_pretraining=True
    )
    
    # Example sentences (already tokenized)
    sentences_a = [
        np.array([2054, 2003, 1996, 2307]),  # "this is the test"
        np.array([2023, 2003, 1037, 8331])    # "it is a example"
    ]
    
    sentences_b = [
        np.array([2009, 2064, 2204, 2171]),  # "we will learn today"
        np.array([1045, 2293, 1037, 3046])    # "i like a cat"
    ]
    
    # NSP labels: 1 = IsNext, 0 = NotNext
    is_next_labels = np.array([1, 0])
    
    # Create NSP data
    token_ids, segment_ids, attention_mask, nsp_labels = create_nsp_data(
        sentences_a, sentences_b, is_next_labels
    )
    
    # Create MLM data
    masked_input, mlm_labels = create_mlm_data(token_ids, mask_prob=0.15, vocab_size=vocab_size)
    
    # Forward pass
    outputs = model.forward(masked_input, position_ids=None, attention_mask=attention_mask)
    
    # Access the different outputs
    mlm_logits = outputs['mlm_logits']
    nsp_logits = outputs['nsp_logits']
    
    print(f"Input shape: {masked_input.shape}")
    print(f"MLM logits shape: {mlm_logits.shape}")
    print(f"NSP logits shape: {nsp_logits.shape}")
    print(f"MLM labels shape: {mlm_labels.shape}")
    print(f"NSP labels shape: {nsp_labels.shape}")
    
    # In a real pretraining scenario, you would:
    # 1. Calculate MLM loss (cross entropy on masked positions only)
    # 2. Calculate NSP loss (binary cross entropy)
    # 3. Combine losses and perform backpropagation
    # 4. Update model parameters
    
    return model

if __name__ == "__main__":
    print("BERT Pretraining Example")
    model = pretraining_example()
    print("\nPretraining example completed successfully!")