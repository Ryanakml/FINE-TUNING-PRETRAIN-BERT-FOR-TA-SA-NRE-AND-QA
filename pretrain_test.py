import numpy as np
from improved_code import BertModel

class BertPretrainingHeads:
    """
    Heads for the two pretraining tasks:
    1. Masked Language Modeling (MLM)
    2. Next Sentence Prediction (NSP)
    """
    def __init__(self, bert_model, vocab_size):
        self.bert = bert_model
        self.d_model = bert_model.word_embedding.d_model
        self.vocab_size = vocab_size
        
        # MLM prediction head
        scale = np.sqrt(2.0 / self.d_model)
        self.mlm_dense = np.random.randn(self.d_model, self.d_model) * scale
        self.mlm_bias = np.zeros((self.d_model,))
        self.mlm_decoder = np.random.randn(self.d_model, vocab_size) * scale
        self.mlm_decoder_bias = np.zeros((vocab_size,))
        
        # NSP prediction head
        self.nsp_dense = np.random.randn(self.d_model, 2) * scale  # Binary classification
        self.nsp_bias = np.zeros((2,))
    
    def gelu(self, x):
        # GELU activation function
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def forward(self, token_ids, segment_ids=None, position_ids=None, attention_mask=None):
        """
        Forward pass for pretraining
        
        Args:
            token_ids: [batch_size, seq_len] Token IDs
            segment_ids: [batch_size, seq_len] Segment IDs (0 for first sentence, 1 for second)
            position_ids: [batch_size, seq_len] Position IDs
            attention_mask: [batch_size, seq_len] Attention mask (1 for tokens to attend to, 0 for padding)
            
        Returns:
            mlm_logits: [batch_size, seq_len, vocab_size] MLM logits
            nsp_logits: [batch_size, 2] NSP logits
        """
        # Get BERT outputs
        bert_outputs = self.bert.forward(token_ids, position_ids)
        
        # MLM task
        mlm_hidden = np.matmul(bert_outputs, self.mlm_dense) + self.mlm_bias
        mlm_hidden = self.gelu(mlm_hidden)
        mlm_logits = np.matmul(mlm_hidden, self.mlm_decoder) + self.mlm_decoder_bias
        
        # NSP task - use [CLS] token (first token)
        cls_output = bert_outputs[:, 0, :]  # [batch_size, d_model]
        nsp_logits = np.matmul(cls_output, self.nsp_dense) + self.nsp_bias
        
        return mlm_logits, nsp_logits


def create_mlm_data(tokens, mask_prob=0.15):
    """
    Create masked input and labels for masked language modeling.
    
    Args:
        tokens: [batch_size, seq_len] Token IDs
        mask_prob: Probability of masking a token
        
    Returns:
        masked_tokens: [batch_size, seq_len] Masked token IDs
        mlm_labels: [batch_size, seq_len] Labels (-1 for unmasked tokens, original token ID for masked)
    """
    masked_tokens = tokens.copy()
    mlm_labels = np.ones_like(tokens) * -1  # -1 for tokens we don't need to predict
    
    # Create mask indices
    prob_matrix = np.random.random(tokens.shape)
    mask_indices = prob_matrix < mask_prob
    
    # Don't mask [CLS], [SEP] or padding tokens (0)
    special_tokens = (tokens == 0) | (tokens == 101) | (tokens == 102)  # Replace with your special token IDs
    mask_indices = mask_indices & ~special_tokens
    
    # Set labels for masked tokens
    mlm_labels[mask_indices] = tokens[mask_indices]
    
    # 80% of the time, replace with [MASK] token
    indices_mask = np.random.random(tokens.shape) < 0.8
    indices_to_mask = mask_indices & indices_mask
    masked_tokens[indices_to_mask] = 103  # [MASK] token ID - replace with your mask token ID
    
    # 10% of the time, replace with random word
    indices_random = np.random.random(tokens.shape) < 0.1
    indices_to_random = mask_indices & ~indices_to_mask & indices_random
    random_words = np.random.randint(1, 30000, size=tokens.shape)  # Replace with your vocab size
    masked_tokens[indices_to_random] = random_words[indices_to_random]
    
    # 10% of the time, keep original
    # (the remaining masked tokens will be kept unchanged)
    
    return masked_tokens, mlm_labels


def create_nsp_data(text_corpus, tokenizer, max_seq_length=512, batch_size=32):
    """
    Create data for Next Sentence Prediction task.
    
    Note: This is a simplified version - in practice, you would use a real tokenizer
    and process actual text from a corpus.
    
    Returns:
        token_ids: [batch_size, seq_len] Token IDs
        segment_ids: [batch_size, seq_len] Segment IDs
        nsp_labels: [batch_size] NSP labels (0=next sentence, 1=random sentence)
    """
    # Simulate tokenized sentences
    # In a real scenario, you would:
    # 1. Select sentence pairs from your corpus
    # 2. For 50% of pairs, select the actual next sentence
    # 3. For 50% of pairs, select a random sentence from the corpus
    
    token_ids = np.zeros((batch_size, max_seq_length), dtype=np.int32)
    segment_ids = np.zeros((batch_size, max_seq_length), dtype=np.int32)
    nsp_labels = np.zeros(batch_size, dtype=np.int32)
    
    for i in range(batch_size):
        # Decide if we use the actual next sentence or a random one
        is_random_next = np.random.random() < 0.5
        nsp_labels[i] = 1 if is_random_next else 0
        
        # Simulate sentence lengths (would come from actual tokenized text)
        # In real implementation, get actual sentences from corpus
        first_len = np.random.randint(10, 128)
        second_len = np.random.randint(10, max_seq_length - first_len - 3)  # Leave room for [CLS], [SEP], [SEP]
        
        # Create tokens - this simulates tokenized text
        token_ids[i, 0] = 101  # [CLS]
        token_ids[i, 1:first_len+1] = np.random.randint(1000, 10000, size=first_len)  # First sentence
        token_ids[i, first_len+1] = 102  # [SEP]
        token_ids[i, first_len+2:first_len+2+second_len] = np.random.randint(1000, 10000, size=second_len)  # Second sentence
        token_ids[i, first_len+2+second_len] = 102  # [SEP]
        
        # Segment IDs
        segment_ids[i, first_len+2:first_len+2+second_len+1] = 1  # Second sentence has segment ID 1
    
    return token_ids, segment_ids, nsp_labels


def cross_entropy_loss(logits, labels, ignore_index=-1):
    """
    Simple cross entropy loss
    
    Args:
        logits: [batch_size, ..., num_classes]
        labels: [batch_size, ...] with values 0 to num_classes-1 or ignore_index
        ignore_index: Value in labels to ignore
        
    Returns:
        loss: Scalar loss value
    """
    # Convert logits to probabilities with softmax
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    
    # Get the probability of the correct class
    batch_size = labels.shape[0]
    if len(logits.shape) == 3:  # For MLM task
        seq_len = labels.shape[1]
        flat_labels = labels.reshape(-1)
        flat_probs = probs.reshape(-1, probs.shape[-1])
        
        # Create mask for non-ignored indices
        mask = (flat_labels != ignore_index)
        valid_labels = flat_labels[mask]
        valid_probs = flat_probs[mask]
        
        # Get the log probability of the correct class for valid labels
        correct_log_probs = -np.log(valid_probs[np.arange(len(valid_labels)), valid_labels])
        
        # Average loss
        return np.mean(correct_log_probs)
    else:  # For NSP task
        correct_log_probs = -np.log(probs[np.arange(batch_size), labels])
        return np.mean(correct_log_probs)


def train_step(model, optimizer, token_ids, segment_ids, masked_tokens, mlm_labels, nsp_labels):
    """
    A single training step
    
    Note: This is simplified - in a real implementation, you would:
    1. Use automatic differentiation (like PyTorch or TensorFlow)
    2. Calculate gradients and update weights properly
    
    Args:
        model: BertPretrainingHeads
        optimizer: Optimizer object
        token_ids: Original token IDs
        segment_ids: Segment IDs
        masked_tokens: Masked token IDs for MLM task
        mlm_labels: MLM labels
        nsp_labels: NSP labels
        
    Returns:
        mlm_loss: MLM loss value
        nsp_loss: NSP loss value
    """
    # Forward pass
    mlm_logits, nsp_logits = model.forward(masked_tokens, segment_ids)
    
    # Calculate losses
    mlm_loss = cross_entropy_loss(mlm_logits, mlm_labels, ignore_index=-1)
    nsp_loss = cross_entropy_loss(nsp_logits, nsp_labels)
    
    # Combined loss
    loss = mlm_loss + nsp_loss
    
    # Here would be the optimizer step (backpropagation and weight updates)
    # For a numpy implementation, you would need to manually compute gradients
    # and update weights, which is quite complex.
    
    return mlm_loss, nsp_loss


def pretrain_bert(model, epochs, batch_size, data_generator, learning_rate=1e-4):
    """
    Pretrain the BERT model
    
    Args:
        model: BertPretrainingHeads
        epochs: Number of epochs to train
        batch_size: Batch size
        data_generator: Function that yields batches of pretraining data
        learning_rate: Learning rate
        
    Returns:
        training_history: Dictionary with training metrics
    """
    history = {
        'mlm_loss': [],
        'nsp_loss': []
    }
    
    # In a real implementation, you would initialize an optimizer here
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        epoch_mlm_loss = 0
        epoch_nsp_loss = 0
        num_batches = 0
        
        # In a real implementation, this would iterate through actual batches from your dataset
        for _ in range(10):  # Simulate 10 batches per epoch
            # Get a batch of data
            token_ids, segment_ids, nsp_labels = create_nsp_data(None, None, max_seq_length=128, batch_size=batch_size)
            masked_tokens, mlm_labels = create_mlm_data(token_ids)
            
            # Train step
            mlm_loss, nsp_loss = train_step(model, None, token_ids, segment_ids, masked_tokens, mlm_labels, nsp_labels)
            
            epoch_mlm_loss += mlm_loss
            epoch_nsp_loss += nsp_loss
            num_batches += 1
            
        # Average losses for the epoch
        epoch_mlm_loss /= num_batches
        epoch_nsp_loss /= num_batches
        
        history['mlm_loss'].append(epoch_mlm_loss)
        history['nsp_loss'].append(epoch_nsp_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - MLM Loss: {epoch_mlm_loss:.4f}, NSP Loss: {epoch_nsp_loss:.4f}")
    
    return history


if __name__ == "__main__":
    # Example pretraining setup with our BERT model
    vocab_size = 30000
    max_seq_length = 512
    d_model = 128  # Smaller for example
    num_heads = 4
    d_ff = 512
    num_layers = 2
    
    # Create base BERT model
    bert_model = BertModel(vocab_size, max_seq_length, d_model, num_heads, d_ff, num_layers)
    
    # Create pretraining model
    pretraining_model = BertPretrainingHeads(bert_model, vocab_size)
    
    # Example pretraining
    print("Starting pretraining simulation...")
    history = pretrain_bert(pretraining_model, epochs=3, batch_size=8, data_generator=None)
    print("Pretraining simulation complete!")
    
    print("\nIn a real implementation, you would:")
    print("1. Use a framework like PyTorch or TensorFlow for automatic differentiation")
    print("2. Use a proper data pipeline with actual text corpus")
    print("3. Use a tokenizer library to process text")
    print("4. Train on multiple GPUs for days or weeks")
    print("5. Save model checkpoints during training")