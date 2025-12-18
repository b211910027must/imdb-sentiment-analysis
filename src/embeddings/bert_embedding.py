"""BERT Embedding Implementation"""
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm


class BERTEmbedding:
    """
    BERT (Bidirectional Encoder Representations from Transformers) Embedding
    
    Mathematical Foundation:
    
    1. Self-Attention Mechanism:
       Attention(Q, K, V) = softmax(QK^T / √d_k)V
       Where Q=query, K=key, V=value, d_k=dimension of keys
    
    2. Multi-Head Attention:
       MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
       head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    3. Pre-training Tasks:
       - MLM (Masked Language Model): Predict masked tokens
       - NSP (Next Sentence Prediction): Predict if sentences are consecutive
    
    Advantages:
    - Bidirectional context understanding
    - Transfer learning from large-scale pre-training
    - Captures semantic relationships
    
    Reference: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
    Paper: arXiv:1810.04805
    """
    
    def __init__(self, model_name="distilbert-base-uncased", max_length=256, batch_size=8, device=None):
        """
        Initialize BERT model.
        
        Args:
            model_name: Pre-trained model name
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            device: Device to use (cpu/cuda/mps)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device if device else torch.device("cpu")
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def encode_batch(self, texts):
        """
        Encode a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Batch of embeddings
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token embedding (first token) - represents entire sequence
            # [CLS] is trained to aggregate semantic information of the whole sentence
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings
    
    def transform(self, texts):
        """
        Transform texts to BERT embeddings.
        
        Args:
            texts: List of text documents
            
        Returns:
            numpy array of document embeddings
        """
        print(f"Generating BERT embeddings for {len(texts)} documents...")
        
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def fit_transform(self, texts):
        """
        Transform texts (no fitting needed for pre-trained BERT).
        
        Args:
            texts: List of text documents
            
        Returns:
            numpy array of document embeddings
        """
        return self.transform(texts)
