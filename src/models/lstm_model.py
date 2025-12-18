"""LSTM Model"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class LSTMClassifier(nn.Module):
    """
    LSTM (Long Short-Term Memory) Neural Network
    
    Mathematical Foundation:
    
    LSTM solves the vanishing gradient problem in RNNs through gating mechanisms:
    
    1. Forget Gate (what to forget from cell state):
       f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
    
    2. Input Gate (what new information to store):
       i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
       C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
    
    3. Cell State Update:
       C_t = f_t * C_{t-1} + i_t * C̃_t
    
    4. Output Gate (what to output):
       o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
       h_t = o_t * tanh(C_t)
    
    Where:
    - σ = sigmoid function
    - * = element-wise multiplication
    - h_t = hidden state
    - C_t = cell state
    - x_t = input at time t
    
    Key Advantage:
    - Cell state (C_t) acts as a "memory highway"
    - Gradients can flow through time without vanishing
    - Selective forgetting and updating through gates
    
    Reference: Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
    Neural Computation 9(8): 1735-1780
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.3):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Size of input features
            hidden_size: Number of LSTM units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Only add dropout parameter if num_layers > 1 (LSTM requirement)
        lstm_kwargs = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'batch_first': True
        }
        if num_layers > 1:
            lstm_kwargs['dropout'] = dropout
        
        self.lstm = nn.LSTM(**lstm_kwargs)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass"""
        # x shape: (batch_size, seq_len, input_size)
        # For our case: (batch_size, 1, input_size) since we use averaged embeddings
        
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Dropout and fully connected layer
        out = self.dropout(last_output)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out


class LSTMModel:
    """Wrapper class for LSTM training and evaluation"""
    
    def __init__(self, hidden_size=64, num_layers=1, dropout=0.3, 
                 batch_size=32, epochs=10, learning_rate=0.001,
                 cv_splits=5, cv_repeats=4, device=None, random_state=42):
        """
        Initialize LSTM model wrapper.
        
        Args:
            hidden_size: Number of LSTM units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate
            cv_splits: Number of CV splits
            cv_repeats: Number of CV repeats
            device: Device to use
            random_state: Random seed
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.device = device if device else torch.device("cpu")
        self.random_state = random_state
        self.model = None
        
    def _train_single_fold(self, X_train, y_train, X_val, y_val):
        """Train model on a single fold"""
        input_size = X_train.shape[1]
        
        # Initialize model
        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).unsqueeze(1),  # Add sequence dimension
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1).to(self.device)
            val_outputs = model(X_val_tensor)
            val_preds = (val_outputs.cpu().numpy() > 0.5).astype(int).flatten()
            val_accuracy = accuracy_score(y_val, val_preds)
        
        return model, val_accuracy
    
    def train(self, X_train, y_train):
        """
        Train model with cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of cross-validation scores
        """
        print(f"Training LSTM with {self.cv_splits}-fold CV × {self.cv_repeats} repeats")
        
        # Convert to numpy if sparse
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        
        # Create cross-validator
        cv = RepeatedStratifiedKFold(
            n_splits=self.cv_splits,
            n_repeats=self.cv_repeats,
            random_state=self.random_state
        )
        
        cv_scores = []
        best_score = 0
        best_model = None
        
        # Cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            model, val_accuracy = self._train_single_fold(
                X_fold_train, y_fold_train, X_fold_val, y_fold_val
            )
            
            cv_scores.append(val_accuracy)
            
            if val_accuracy > best_score:
                best_score = val_accuracy
                best_model = model
            
            if (fold_idx + 1) % 5 == 0:
                print(f"Completed {fold_idx + 1}/{self.cv_splits * self.cv_repeats} folds, "
                      f"Current fold accuracy: {val_accuracy:.4f}")
        
        self.model = best_model
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        print(f"CV Score: {mean_score:.4f} ± {std_score:.4f}")
        
        return {
            'best_score': mean_score,
            'cv_scores': cv_scores,
            'std_score': std_score
        }
    
    def predict(self, X):
        """Make predictions"""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
            outputs = self.model(X_tensor)
            predictions = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
        
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)
            outputs = self.model(X_tensor).cpu().numpy()
            # Return probabilities for both classes
            proba = np.hstack([1 - outputs, outputs])
        
        return proba
