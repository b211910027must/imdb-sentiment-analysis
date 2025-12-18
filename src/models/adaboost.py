"""AdaBoost Model"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
import numpy as np


class AdaBoostModel:
    """
    AdaBoost (Adaptive Boosting) Classifier
    
    Mathematical Foundation:
    
    1. Initialize weights:
       w_i^(1) = 1/N for all training samples
    
    2. For each iteration t = 1 to T:
       
       a) Train weak classifier h_t with weights w^(t)
       
       b) Compute weighted error:
          ε_t = Σ w_i^(t) * I(h_t(x_i) ≠ y_i) / Σ w_i^(t)
       
       c) Compute classifier weight:
          α_t = 0.5 * ln((1 - ε_t) / ε_t)
       
       d) Update sample weights:
          w_i^(t+1) = w_i^(t) * exp(-α_t * y_i * h_t(x_i))
          Then normalize: w_i^(t+1) = w_i^(t+1) / Σ w_j^(t+1)
    
    3. Final hypothesis:
       H(x) = sign(Σ α_t * h_t(x))
    
    Key Insight:
    - Focuses on hard-to-classify samples by increasing their weights
    - Combines weak learners into a strong learner
    - Each classifier weight (α_t) depends on its accuracy
    
    Advantages:
    - Simple and effective
    - Less prone to overfitting than other boosting methods
    - No hyperparameter tuning required (though recommended)
    
    Reference: Freund & Schapire, "A Decision-Theoretic Generalization of 
    On-Line Learning and an Application to Boosting" (1996)
    Journal of Computer and System Sciences 55(1): 119-139
    """
    
    def __init__(self, param_grid=None, cv_splits=5, cv_repeats=4, random_state=42):
        """
        Initialize AdaBoost with hyperparameter tuning.
        
        Args:
            param_grid: Dictionary of hyperparameters to search
            cv_splits: Number of CV splits
            cv_repeats: Number of CV repeats
            random_state: Random seed
        """
        self.param_grid = param_grid or {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.5, 1.0, 1.5]
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
        print(f"Training AdaBoost with {self.cv_splits}-fold CV × {self.cv_repeats} repeats")
        
        # Create cross-validator
        cv = RepeatedStratifiedKFold(
            n_splits=self.cv_splits,
            n_repeats=self.cv_repeats,
            random_state=self.random_state
        )
        
        # Grid search
        base_model = AdaBoostClassifier(random_state=self.random_state, algorithm='SAMME')
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
