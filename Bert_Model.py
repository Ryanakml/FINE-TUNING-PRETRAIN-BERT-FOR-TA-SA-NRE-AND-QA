import numpy as np

# Matrix multiplication function - using np.matmul directly
# Note: np.dot doesn't handle batched matrix multiplication correctly in all cases

# Softmax Activation Function
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# Normalization Layer Function
def norm(x, eps=1e-6):
    avg = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - avg) / (std + eps)

# Word Embedding Lookup Table
class WordEmbedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Initialize with Xavier/Glorot initialization for better convergence
        self.embedding_table = np.random.randn(vocab_size, d_model) * np.sqrt(2.0 / (vocab_size + d_model))

    def forward(self, token_ids):
        return self.embedding_table[token_ids]

# Positional Encoding Lookup table
class PositionalEncoding:
    def __init__(self, max_seq, d_model):
        self.max_seq = max_seq
        self.d_model = d_model
        
        # Standard sinusoidal positional encoding from "Attention is All You Need"
        position = np.arange(max_seq)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_seq, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.embedding_table = pe

    def forward(self, position_ids):
        return self.embedding_table[position_ids]

# Self Attention
class SelfAttention:
    def __init__(self, d_model):
        self.d_model = d_model
        # Better initialization
        scale = np.sqrt(2.0 / d_model)
        self.w_q = np.random.randn(d_model, d_model) * scale
        self.w_k = np.random.randn(d_model, d_model) * scale
        self.w_v = np.random.randn(d_model, d_model) * scale

    def forward(self, x, attention_mask=None):
        # Using np.matmul for consistent matrix multiplication
        q = np.matmul(x, self.w_q)
        k = np.matmul(x, self.w_k)
        v = np.matmul(x, self.w_v)

        # Proper handling of batch matrix multiplication
        # For shape (batch_size, seq_len, d_model)
        # Transpose k for matrix multiplication
        k_t = np.transpose(k, (0, 2, 1))
        
        attn_scores = np.matmul(q, k_t) / np.sqrt(self.d_model)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Reshape mask for broadcasting: [batch_size, 1, seq_len]
            mask = attention_mask[:, np.newaxis, :]
            # Apply large negative value to masked positions before softmax
            attn_scores = attn_scores * mask - 10000.0 * (1.0 - mask)
            
        attn_probs = softmax(attn_scores, axis=-1)  # Ensure softmax along correct axis
        attn_output = np.matmul(attn_probs, v)

        return attn_output

# Multihead Attention
class MultiheadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Better initialization
        scale = np.sqrt(2.0 / d_model)
        self.w_q = np.random.randn(d_model, d_model) * scale
        self.w_k = np.random.randn(d_model, d_model) * scale
        self.w_v = np.random.randn(d_model, d_model) * scale
        self.w_o = np.random.randn(d_model, d_model) * scale

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape

        # Q, K, V
        q = np.matmul(x, self.w_q)  # Using np.matmul directly for clarity
        k = np.matmul(x, self.w_k)
        v = np.matmul(x, self.w_v)

        # Split into multiple heads - carefully reshape to ensure dimensions are correct
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to get shape (batch_size, num_heads, seq_len, head_dim)
        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))

        # Self attention per head
        # Transpose k for matrix multiplication
        k_t = np.transpose(k, (0, 1, 3, 2))
        
        # Calculate attention scores
        attn_scores = np.matmul(q, k_t) / np.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Reshape mask for broadcasting: [batch_size, 1, 1, seq_len]
            mask = attention_mask[:, np.newaxis, np.newaxis, :]
            # Apply large negative value to masked positions before softmax
            attn_scores = attn_scores * mask - 10000.0 * (1.0 - mask)
            
        attn_probs = softmax(attn_scores, axis=-1)
        attn_output = np.matmul(attn_probs, v)

        # Transpose back and reshape to combine heads
        attn_output = np.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)

        # Linear projection 
        output = np.matmul(attn_output, self.w_o)

        return output

# Feed Forward Neural Network
class FeedForward:
    def __init__(self, d_model, d_ff):
        # Better initialization
        scale_1 = np.sqrt(2.0 / d_model)
        scale_2 = np.sqrt(2.0 / d_ff)
        
        self.w1 = np.random.randn(d_model, d_ff) * scale_1
        self.b1 = np.zeros((d_ff,))
        self.w2 = np.random.randn(d_ff, d_model) * scale_2
        self.b2 = np.zeros((d_model,))

    def gelu(self, x):
        # More modern activation function used in BERT
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def forward(self, x):
        # Handle batched matrix multiplication
        x_expanded = np.matmul(x, self.w1) + self.b1
        x_activated = self.gelu(x_expanded)  # Using GELU instead of ReLU
        output = np.matmul(x_activated, self.w2) + self.b2
        return output

# BERT Block
class BertBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiheadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
    
    def forward(self, x, attention_mask=None):
        # Multihead self attention
        x_mha = self.mha.forward(x)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match attention dimensions
            # attention_mask should be [batch_size, seq_len]
            # We need to reshape it for broadcasting
            mask = attention_mask[:, :, np.newaxis]
            x_mha = x_mha * mask
        
        # Add and norm
        x_norm1 = norm(x + x_mha)
        
        # Feed forward
        x_ffn = self.ff.forward(x_norm1)
        
        # Add and norm
        x_norm2 = norm(x_norm1 + x_ffn)
        
        return x_norm2

# BERT Model
class BertModel:
    def __init__(self, vocab_size, max_seq_length, d_model, num_heads, d_ff, num_layers):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.word_embedding = WordEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEncoding(max_seq_length, d_model)
        self.layers = [BertBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
    
    def forward(self, token_ids, position_ids=None, attention_mask=None):
        # If position_ids not provided, create sequential ones
        batch_size = token_ids.shape[0]
        seq_len = token_ids.shape[1]
        
        if position_ids is None:
            # Create position IDs for each item in the batch
            position_ids = np.tile(np.arange(seq_len), (batch_size, 1))
            
        # Get embeddings
        word_emb = self.word_embedding.forward(token_ids)
        pos_emb = self.position_embedding.forward(position_ids)
        
        # Combined embeddings
        x = word_emb + pos_emb
        
        # Process through all layers
        for layer in self.layers:
            x = layer.forward(x, attention_mask)
            
        return x

# Masked Language Model Head
class MaskedLMHead:
    def __init__(self, d_model, vocab_size):
        # Initialize with better weights
        scale = np.sqrt(2.0 / d_model)
        self.dense = np.random.randn(d_model, d_model) * scale
        self.dense_bias = np.zeros((d_model,))
        self.layer_norm = True  # Flag to apply layer normalization
        self.decoder = np.random.randn(d_model, vocab_size) * scale
        self.decoder_bias = np.zeros((vocab_size,))
        
    def forward(self, hidden_states):
        # Use the hidden states from the BERT model
        x = np.matmul(hidden_states, self.dense) + self.dense_bias
        x = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))  # GELU
        
        if self.layer_norm:
            x = norm(x)
            
        # Project back to vocabulary size
        logits = np.matmul(x, self.decoder) + self.decoder_bias
        return logits

# Next Sentence Prediction Head
class NSPHead:
    def __init__(self, d_model):
        scale = np.sqrt(2.0 / d_model)
        self.dense = np.random.randn(d_model, 2) * scale  # Binary classification
        self.dense_bias = np.zeros((2,))
        
    def forward(self, pooled_output):
        # Use the [CLS] token representation (first token)
        logits = np.matmul(pooled_output, self.dense) + self.dense_bias
        return logits

# BERT For Pretraining
class BertForPretraining:
    def __init__(self, vocab_size, max_seq_length, d_model, num_heads, d_ff, num_layers):
        self.bert = BertModel(vocab_size, max_seq_length, d_model, num_heads, d_ff, num_layers)
        self.mlm_head = MaskedLMHead(d_model, vocab_size)
        self.nsp_head = NSPHead(d_model)
        
    def forward(self, token_ids, position_ids=None, attention_mask=None):
        # Get sequence output from base BERT model
        sequence_output = self.bert.forward(token_ids, position_ids, attention_mask)
        
        # Get the first token ([CLS]) for NSP task
        pooled_output = sequence_output[:, 0, :]
        
        # Get MLM and NSP predictions
        mlm_logits = self.mlm_head.forward(sequence_output)
        nsp_logits = self.nsp_head.forward(pooled_output)
        
        return {
            'mlm_logits': mlm_logits,  # Shape: [batch_size, seq_len, vocab_size]
            'nsp_logits': nsp_logits,   # Shape: [batch_size, 2]
            'sequence_output': sequence_output,  # Full output for other tasks
            'pooled_output': pooled_output      # [CLS] token output
        }

# Default configuration function to easily create models with standard parameters
def create_bert_model(vocab_size=30000, max_seq_length=512, d_model=768, num_heads=12, 
                     d_ff=3072, num_layers=12, for_pretraining=True):
    """Create a BERT model with standard or custom parameters.
    
    Args:
        vocab_size: Size of the vocabulary
        max_seq_length: Maximum sequence length
        d_model: Hidden size of the model
        num_heads: Number of attention heads
        d_ff: Size of the intermediate feed-forward layer
        num_layers: Number of transformer layers
        for_pretraining: Whether to return a model with MLM and NSP heads
        
    Returns:
        A BertModel or BertForPretraining instance
    """
    if for_pretraining:
        return BertForPretraining(vocab_size, max_seq_length, d_model, num_heads, d_ff, num_layers)
    else:
        return BertModel(vocab_size, max_seq_length, d_model, num_heads, d_ff, num_layers)