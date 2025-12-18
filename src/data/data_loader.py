"""Data loading and preprocessing for IMDB dataset"""
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class IMDBDataLoader:
    """Load and preprocess IMDB movie reviews dataset"""
    
    def __init__(self, dataset_path, max_samples=None, test_size=0.2, random_state=42):
        """
        Initialize data loader.
        
        Args:
            dataset_path: Path to IMDB CSV file
            max_samples: Maximum number of samples to load (None for all)
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.max_samples = max_samples
        self.test_size = test_size
        self.random_state = random_state
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """
        Clean and preprocess text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_remove_stopwords(self, text):
        """
        Tokenize text and remove stopwords.
        
        Args:
            text: Cleaned text string
            
        Returns:
            List of tokens
        """
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        return tokens
    
    def load_data(self, remove_stopwords=False):
        """
        Load and preprocess IMDB dataset.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"Loading dataset from {self.dataset_path}...")
        
        # Load CSV
        df = pd.read_csv(self.dataset_path)
        
        # Limit samples if specified
        if self.max_samples and self.max_samples < len(df):
            df = df.sample(n=self.max_samples, random_state=self.random_state)
            print(f"Sampled {self.max_samples} reviews")
        
        # Clean text
        print("Cleaning text...")
        df['clean_review'] = df['review'].apply(self.clean_text)
        
        # Optionally remove stopwords
        if remove_stopwords:
            print("Removing stopwords...")
            df['tokens'] = df['clean_review'].apply(self.tokenize_and_remove_stopwords)
            df['clean_review'] = df['tokens'].apply(lambda x: ' '.join(x))
        
        # Convert sentiment to binary (0: negative, 1: positive)
        df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})
        
        # Split data
        X = df['clean_review'].values
        y = df['sentiment'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Positive samples: {np.sum(y == 1)}")
        print(f"Negative samples: {np.sum(y == 0)}")
        
        return X_train, X_test, y_train, y_test
