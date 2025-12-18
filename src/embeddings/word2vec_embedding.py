"""Word2Vec Embedding Implementation"""
from gensim.models import Word2Vec
import numpy as np


class Word2VecEmbedding:
    """
    Word2Vec Embedding (CBOW and Skip-gram)
    
    Mathematical Foundation:
    
    1. CBOW (Continuous Bag of Words):
       Predicts target word from context words
       Objective: maximize P(w_t | w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c})
       
    2. Skip-gram:
       Predicts context words from target word
       Objective: maximize P(w_{t+j} | w_t) for j in [-c, c], j ≠ 0
    
    Training uses negative sampling to efficiently compute softmax:
    log σ(v'_{w_O}^T v_{w_I}) + Σ E_{w_i ~ P_n}[log σ(-v'_{w_i}^T v_{w_I})]
    
    Reference: Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013)
    Paper: arXiv:1301.3781
    """
    
    def __init__(self, vector_size=100, window=5, min_count=2, workers=4, epochs=10, sg=0):
        """
        Initialize Word2Vec model.
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Context window size
            min_count: Minimum word frequency
            workers: Number of worker threads
            epochs: Number of training epochs
            sg: Training algorithm (0=CBOW, 1=Skip-gram)
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.sg = sg
        self.model = None
        self.model_type = "Skip-gram" if sg == 1 else "CBOW"
        
    def fit(self, texts):
        """
        Train Word2Vec model.
        
        Args:
            texts: List of text documents
        """
        # Tokenize texts
        tokenized_texts = [text.split() for text in texts]
        
        print(f"Training Word2Vec ({self.model_type}) with vector_size={self.vector_size}, "
              f"window={self.window}, epochs={self.epochs}")
        
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            sg=self.sg
        )
        
        print(f"Vocabulary size: {len(self.model.wv)}")
        
    def transform(self, texts):
        """
        Transform texts to average word vectors.
        
        Args:
            texts: List of text documents
            
        Returns:
            numpy array of document vectors
        """
        vectors = []
        for text in texts:
            words = text.split()
            word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]
            
            if word_vectors:
                # Average word vectors
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                # Zero vector for documents with no known words
                doc_vector = np.zeros(self.vector_size)
            
            vectors.append(doc_vector)
        
        return np.array(vectors)
    
    def fit_transform(self, texts):
        """
        Train model and transform texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            numpy array of document vectors
        """
        self.fit(texts)
        return self.transform(texts)
