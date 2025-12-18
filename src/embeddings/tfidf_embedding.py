"""TF-IDF Embedding Implementation"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TFIDFEmbedding:
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) Embedding
    
    Mathematical Foundation:
    TF-IDF(t, d) = TF(t, d) × IDF(t)
    
    Where:
    - TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in d)
    - IDF(t) = log(Total number of documents / Number of documents containing term t)
    
    This weights terms by their importance in a document relative to the entire corpus.
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams to consider
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True  # Apply sublinear tf scaling (1 + log(tf))
        )
        
    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        print(f"Fitting TF-IDF with max_features={self.max_features}, ngram_range={self.ngram_range}")
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """
        Transform texts using fitted vectorizer.
        
        Args:
            texts: List of text documents
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()
