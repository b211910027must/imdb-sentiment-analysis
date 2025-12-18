"""Main experiment runner for IMDB sentiment analysis"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import json

from src.utils.device_utils import get_device
from src.data.data_loader import IMDBDataLoader
from src.embeddings.tfidf_embedding import TFIDFEmbedding
from src.embeddings.word2vec_embedding import Word2VecEmbedding
from src.embeddings.bert_embedding import BERTEmbedding
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.adaboost import AdaBoostModel
from src.models.lstm_model import LSTMModel
from src.evaluation.metrics import Evaluator


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_experiment(embedding_name, embedding, model_name, model, X_train, X_test, y_train, y_test):
    """
    Run a single experiment.
    
    Args:
        embedding_name: Name of the embedding method
        embedding: Embedding object
        model_name: Name of the model
        model: Model object
        X_train, X_test, y_train, y_test: Train/test data
        
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*80}")
    print(f"Running: {embedding_name} + {model_name}")
    print(f"{'='*80}")
    
    try:
        # Train model
        cv_results = model.train(X_train, y_train)
        
        # Test predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Evaluate
        metrics = Evaluator.evaluate(y_test, y_pred, y_proba)
        Evaluator.print_metrics(metrics, f"{embedding_name} + {model_name}")
        
        return {
            'embedding': embedding_name,
            'model': model_name,
            'cv_score': cv_results.get('best_score', 0),
            'test_accuracy': metrics['accuracy'],
            'test_precision': metrics['precision'],
            'test_recall': metrics['recall'],
            'test_f1': metrics['f1'],
            'status': 'success'
        }
    except Exception as e:
        print(f"Error in {embedding_name} + {model_name}: {str(e)}")
        return {
            'embedding': embedding_name,
            'model': model_name,
            'status': 'failed',
            'error': str(e)
        }


def main():
    """Main experiment runner"""
    print("="*80)
    print("IMDB Sentiment Analysis - Embedding & Model Comparison")
    print("="*80)
    
    # Load configuration
    config = load_config()
    
    # Set device
    device = get_device(
        use_mps=config['device']['use_mps'],
        fallback_to_cpu=config['device']['fallback_to_cpu']
    )
    
    # Load data
    data_loader = IMDBDataLoader(
        dataset_path=config['data']['dataset_path'],
        max_samples=config['data']['max_samples'],
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    X_train_raw, X_test_raw, y_train, y_test = data_loader.load_data()
    
    # Store all results
    all_results = []
    
    # ===========================
    # 1. TF-IDF Embeddings
    # ===========================
    print("\n" + "="*80)
    print("Creating TF-IDF Embeddings")
    print("="*80)
    
    tfidf = TFIDFEmbedding(
        max_features=config['embeddings']['tfidf']['max_features'],
        ngram_range=tuple(config['embeddings']['tfidf']['ngram_range'])
    )
    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    X_test_tfidf = tfidf.transform(X_test_raw)
    
    # TF-IDF + Logistic Regression
    lr = LogisticRegressionModel(
        param_grid=config['models']['logistic_regression'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        random_state=config['data']['random_state']
    )
    result = run_experiment("TF-IDF", tfidf, "Logistic Regression", lr,
                           X_train_tfidf, X_test_tfidf, y_train, y_test)
    all_results.append(result)
    
    # TF-IDF + Random Forest
    rf = RandomForestModel(
        param_grid=config['models']['random_forest'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        random_state=config['data']['random_state']
    )
    result = run_experiment("TF-IDF", tfidf, "Random Forest", rf,
                           X_train_tfidf, X_test_tfidf, y_train, y_test)
    all_results.append(result)
    
    # TF-IDF + AdaBoost
    ada = AdaBoostModel(
        param_grid=config['models']['adaboost'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        random_state=config['data']['random_state']
    )
    result = run_experiment("TF-IDF", tfidf, "AdaBoost", ada,
                           X_train_tfidf, X_test_tfidf, y_train, y_test)
    all_results.append(result)
    
    # TF-IDF + LSTM
    lstm = LSTMModel(
        hidden_size=config['models']['lstm']['hidden_size'],
        num_layers=config['models']['lstm']['num_layers'],
        dropout=config['models']['lstm']['dropout'],
        batch_size=config['models']['lstm']['batch_size'],
        epochs=config['models']['lstm']['epochs'],
        learning_rate=config['models']['lstm']['learning_rate'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        device=device,
        random_state=config['data']['random_state']
    )
    result = run_experiment("TF-IDF", tfidf, "LSTM", lstm,
                           X_train_tfidf, X_test_tfidf, y_train, y_test)
    all_results.append(result)
    
    # ===========================
    # 2. Word2Vec CBOW Embeddings
    # ===========================
    print("\n" + "="*80)
    print("Creating Word2Vec (CBOW) Embeddings")
    print("="*80)
    
    w2v_cbow = Word2VecEmbedding(
        vector_size=config['embeddings']['word2vec']['vector_size'],
        window=config['embeddings']['word2vec']['window'],
        min_count=config['embeddings']['word2vec']['min_count'],
        workers=config['embeddings']['word2vec']['workers'],
        epochs=config['embeddings']['word2vec']['epochs'],
        sg=0  # CBOW
    )
    X_train_w2v_cbow = w2v_cbow.fit_transform(X_train_raw)
    X_test_w2v_cbow = w2v_cbow.transform(X_test_raw)
    
    # Word2Vec CBOW + Logistic Regression
    lr = LogisticRegressionModel(
        param_grid=config['models']['logistic_regression'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        random_state=config['data']['random_state']
    )
    result = run_experiment("Word2Vec (CBOW)", w2v_cbow, "Logistic Regression", lr,
                           X_train_w2v_cbow, X_test_w2v_cbow, y_train, y_test)
    all_results.append(result)
    
    # Word2Vec CBOW + Random Forest
    rf = RandomForestModel(
        param_grid=config['models']['random_forest'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        random_state=config['data']['random_state']
    )
    result = run_experiment("Word2Vec (CBOW)", w2v_cbow, "Random Forest", rf,
                           X_train_w2v_cbow, X_test_w2v_cbow, y_train, y_test)
    all_results.append(result)
    
    # Word2Vec CBOW + AdaBoost
    ada = AdaBoostModel(
        param_grid=config['models']['adaboost'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        random_state=config['data']['random_state']
    )
    result = run_experiment("Word2Vec (CBOW)", w2v_cbow, "AdaBoost", ada,
                           X_train_w2v_cbow, X_test_w2v_cbow, y_train, y_test)
    all_results.append(result)
    
    # Word2Vec CBOW + LSTM
    lstm = LSTMModel(
        hidden_size=config['models']['lstm']['hidden_size'],
        num_layers=config['models']['lstm']['num_layers'],
        dropout=config['models']['lstm']['dropout'],
        batch_size=config['models']['lstm']['batch_size'],
        epochs=config['models']['lstm']['epochs'],
        learning_rate=config['models']['lstm']['learning_rate'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        device=device,
        random_state=config['data']['random_state']
    )
    result = run_experiment("Word2Vec (CBOW)", w2v_cbow, "LSTM", lstm,
                           X_train_w2v_cbow, X_test_w2v_cbow, y_train, y_test)
    all_results.append(result)
    
    # ===========================
    # 3. Word2Vec Skip-gram Embeddings
    # ===========================
    print("\n" + "="*80)
    print("Creating Word2Vec (Skip-gram) Embeddings")
    print("="*80)
    
    w2v_sg = Word2VecEmbedding(
        vector_size=config['embeddings']['word2vec']['vector_size'],
        window=config['embeddings']['word2vec']['window'],
        min_count=config['embeddings']['word2vec']['min_count'],
        workers=config['embeddings']['word2vec']['workers'],
        epochs=config['embeddings']['word2vec']['epochs'],
        sg=1  # Skip-gram
    )
    X_train_w2v_sg = w2v_sg.fit_transform(X_train_raw)
    X_test_w2v_sg = w2v_sg.transform(X_test_raw)
    
    # Word2Vec Skip-gram + Logistic Regression
    lr = LogisticRegressionModel(
        param_grid=config['models']['logistic_regression'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        random_state=config['data']['random_state']
    )
    result = run_experiment("Word2Vec (Skip-gram)", w2v_sg, "Logistic Regression", lr,
                           X_train_w2v_sg, X_test_w2v_sg, y_train, y_test)
    all_results.append(result)
    
    # Word2Vec Skip-gram + Random Forest
    rf = RandomForestModel(
        param_grid=config['models']['random_forest'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        random_state=config['data']['random_state']
    )
    result = run_experiment("Word2Vec (Skip-gram)", w2v_sg, "Random Forest", rf,
                           X_train_w2v_sg, X_test_w2v_sg, y_train, y_test)
    all_results.append(result)
    
    # Word2Vec Skip-gram + AdaBoost
    ada = AdaBoostModel(
        param_grid=config['models']['adaboost'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        random_state=config['data']['random_state']
    )
    result = run_experiment("Word2Vec (Skip-gram)", w2v_sg, "AdaBoost", ada,
                           X_train_w2v_sg, X_test_w2v_sg, y_train, y_test)
    all_results.append(result)
    
    # Word2Vec Skip-gram + LSTM
    lstm = LSTMModel(
        hidden_size=config['models']['lstm']['hidden_size'],
        num_layers=config['models']['lstm']['num_layers'],
        dropout=config['models']['lstm']['dropout'],
        batch_size=config['models']['lstm']['batch_size'],
        epochs=config['models']['lstm']['epochs'],
        learning_rate=config['models']['lstm']['learning_rate'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        device=device,
        random_state=config['data']['random_state']
    )
    result = run_experiment("Word2Vec (Skip-gram)", w2v_sg, "LSTM", lstm,
                           X_train_w2v_sg, X_test_w2v_sg, y_train, y_test)
    all_results.append(result)
    
    # ===========================
    # 4. BERT (DistilBERT) Embeddings
    # ===========================
    print("\n" + "="*80)
    print("Creating BERT (DistilBERT) Embeddings")
    print("="*80)
    
    bert = BERTEmbedding(
        model_name=config['embeddings']['bert']['model_name'],
        max_length=config['embeddings']['bert']['max_length'],
        batch_size=config['embeddings']['bert']['batch_size'],
        device=device
    )
    X_train_bert = bert.fit_transform(X_train_raw)
    X_test_bert = bert.transform(X_test_raw)
    
    # BERT + Logistic Regression
    lr = LogisticRegressionModel(
        param_grid=config['models']['logistic_regression'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        random_state=config['data']['random_state']
    )
    result = run_experiment("DistilBERT", bert, "Logistic Regression", lr,
                           X_train_bert, X_test_bert, y_train, y_test)
    all_results.append(result)
    
    # BERT + Random Forest
    rf = RandomForestModel(
        param_grid=config['models']['random_forest'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        random_state=config['data']['random_state']
    )
    result = run_experiment("DistilBERT", bert, "Random Forest", rf,
                           X_train_bert, X_test_bert, y_train, y_test)
    all_results.append(result)
    
    # BERT + AdaBoost
    ada = AdaBoostModel(
        param_grid=config['models']['adaboost'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        random_state=config['data']['random_state']
    )
    result = run_experiment("DistilBERT", bert, "AdaBoost", ada,
                           X_train_bert, X_test_bert, y_train, y_test)
    all_results.append(result)
    
    # BERT + LSTM
    lstm = LSTMModel(
        hidden_size=config['models']['lstm']['hidden_size'],
        num_layers=config['models']['lstm']['num_layers'],
        dropout=config['models']['lstm']['dropout'],
        batch_size=config['models']['lstm']['batch_size'],
        epochs=config['models']['lstm']['epochs'],
        learning_rate=config['models']['lstm']['learning_rate'],
        cv_splits=config['cross_validation']['n_splits'],
        cv_repeats=config['cross_validation']['n_repeats'],
        device=device,
        random_state=config['data']['random_state']
    )
    result = run_experiment("DistilBERT", bert, "LSTM", lstm,
                           X_train_bert, X_test_bert, y_train, y_test)
    all_results.append(result)
    
    # ===========================
    # Save Results
    # ===========================
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    csv_path = f"results/results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Save to JSON
    json_path = f"results/results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {json_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Pivot table for accuracy
    pivot = results_df[results_df['status'] == 'success'].pivot(
        index='embedding',
        columns='model',
        values='test_accuracy'
    )
    
    print("\nTest Accuracy (%):")
    print((pivot * 100).round(2))
    
    print("\n" + "="*80)
    print("Experiments Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
