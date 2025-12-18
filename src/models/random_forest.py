"""Random Forest Model"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
import numpy as np


class RandomForestModel:
    """
    Random Forest Classifier
    
    Mathematical Foundation:
    
    1. Bootstrap Aggregating (Bagging):
       - Create B bootstrap samples from training data
       - Train a decision tree on each sample
       - Aggregate predictions by majority voting
    
    2. Decision Tree Splitting Criteria:
       
       Gini Impurity:
       Gini(p) = 1 - Σ(p_i^2)
       where p_i is the proportion of class i
       
       Information Gain:
       IG(D, A) = Entropy(D) - Σ(|D_v|/|D|) * Entropy(D_v)
       Entropy(D) = -Σ p_i * log_2(p_i)
    
    3. Feature Importance:
       Computed as the total reduction in node impurity
       weighted by probability of reaching that node
    
    4. Final Prediction:
       ŷ = mode{h_1(x), h_2(x), ..., h_B(x)}
    
    Advantages:
    - Reduces overfitting through averaging
    - Handles non-linear relationships
    - Provides feature importance
    - Robust to outliers
    
    Reference: Breiman, "Random Forests" (2001)
    Machine Learning 45(1): 5-32
    """
    
    def __init__(self, param_grid=None, cv_splits=5, cv_repeats=4, random_state=42):
        """
        Initialize Random Forest with hyperparameter tuning.
        
        Args:
            param_grid: Dictionary of hyperparameters to search
            cv_splits: Number of CV splits
            cv_repeats: Number of CV repeats
            random_state: Random seed
        """
        self.param_grid = param_grid or {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
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
        print(f"Training Random Forest with {self.cv_splits}-fold CV × {self.cv_repeats} repeats")
        
        # Create cross-validator
        cv = RepeatedStratifiedKFold(
            n_splits=self.cv_splits,
            n_repeats=self.cv_repeats,
            random_state=self.random_state
        )
        
        # Grid search
        base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
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
    
    def get_feature_importance(self):
        """Get feature importances"""
        return self.model.feature_importances_
