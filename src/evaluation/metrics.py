"""Evaluation metrics for classification models"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


class Evaluator:
    """Evaluate classification models"""
    
    @staticmethod
    def evaluate(y_true, y_pred, y_proba=None):
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary')
        }
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['confusion_matrix'] = {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics, model_name="Model"):
        """
        Print evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
        """
        print(f"\n{'='*50}")
        print(f"{model_name} Evaluation Results")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"TN: {cm['tn']:5d}  FP: {cm['fp']:5d}")
        print(f"FN: {cm['fn']:5d}  TP: {cm['tp']:5d}")
        print(f"{'='*50}\n")
