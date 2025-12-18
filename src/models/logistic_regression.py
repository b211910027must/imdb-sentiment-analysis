"""Logistic Regression Model"""
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
import numpy as np


class LogisticRegressionModel:
    """
    Logistic Regression for Binary Classification
    
    Mathematical Foundation:
    
    1. Sigmoid Function:
       σ(z) = 1 / (1 + e^(-z))
       where z = w^T x + b
    
    2. Hypothesis:
       h_θ(x) = σ(θ^T x) = P(y=1|x;θ)
    
    3. Cost Function (Binary Cross-Entropy):
       J(θ) = -1/m Σ[y*log(h_θ(x)) + (1-y)*log(1-h_θ(x))] + λ||θ||^2
    
    4. Gradient Descent Update:
       θ := θ - α∇J(θ)
    
    5. Regularization:
       - L1 (Lasso): λΣ|θ_j| - promotes sparsity
       - L2 (Ridge): λΣθ_j^2 - prevents overfitting
    
    Advantages:
    - Simple, interpretable, fast
    - Works well with high-dimensional sparse data
    - Probabilistic predictions
    """
    
    def __init__(self, param_grid=None, cv_splits=5, cv_repeats=4, random_state=42):
        """
        Initialize Logistic Regression with hyperparameter tuning.
        
        Args:
            param_grid: Dictionary of hyperparameters to search
            cv_splits: Number of CV splits
            cv_repeats: Number of CV repeats
            random_state: Random seed
        """
        self.param_grid = param_grid or {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'max_iter': [1000]
        }
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.random_state = random_state
        self.model = None
        self.best_params = None
        
    def train(self, X_train, y_train):
        """
        Train model with cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of cross-validation scores
        """
        print(f"Training Logistic Regression with {self.cv_splits}-fold CV × {self.cv_repeats} repeats")
        
        # Create cross-validator
        cv = RepeatedStratifiedKFold(
            n_splits=self.cv_splits,
            n_repeats=self.cv_repeats,
            random_state=self.random_state
        )
        
        # Grid search
        base_model = SKLogisticRegression(random_state=self.random_state)
        grid_search = GridSearchCV(
            base_model,
            self.param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return {
            'best_score': grid_search.best_score_,
            'best_params': self.best_params,
            'cv_results': grid_search.cv_results_
        }
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
