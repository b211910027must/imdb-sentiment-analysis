# 🎬 IMDB Sentiment Analysis - Embedding & Classification Model Comparison

A comprehensive study comparing different text embedding methods (TF-IDF, Word2Vec, BERT) and classification models (Logistic Regression, Random Forest, AdaBoost, LSTM) for sentiment analysis on the IMDB Movie Reviews dataset.

## 📊 Dataset

**IMDB Dataset of 50K Movie Reviews**
- **Source**: [Kaggle - IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size**: 50,000 movie reviews
- **Task**: Binary Sentiment Classification (Positive/Negative)
- **Split**: 25,000 reviews for training, 25,000 for testing (balanced)
- **Format**: CSV with columns: `review`, `sentiment`

## 🎯 Project Overview

This project implements and compares multiple text embedding techniques and machine learning models to analyze sentiment in movie reviews. The goal is to understand which combinations work best and why.

## 🧮 Embedding Methods

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)

**Mathematical Foundation:**

TF-IDF measures the importance of a word in a document relative to a corpus:

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

Where:
- TF(t, d) = (count of term t in document d) / (total terms in d)
- IDF(t) = log(total documents / documents containing term t)
```

**Characteristics:**
- **Advantages**: Simple, fast, interpretable, works well with sparse high-dimensional data
- **Disadvantages**: No semantic understanding, treats words independently (bag-of-words)
- **Use Case**: Baseline method, keyword-based classification

**References:**
- Salton & Buckley, "Term-weighting approaches in automatic text retrieval" (1988)
- Information Processing & Management 24(5): 513-523

### 2. Word2Vec

**Mathematical Foundation:**

Word2Vec learns dense vector representations of words using neural networks.

**a) CBOW (Continuous Bag of Words):**
Predicts the target word from surrounding context words.

```
Objective: maximize P(w_t | w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c})

Loss: L = -log P(w_t | context)
```

**b) Skip-gram:**
Predicts context words from the target word.

```
Objective: maximize Π P(w_{t+j} | w_t) for j ∈ [-c, c], j ≠ 0

Loss: L = -Σ log P(w_{t+j} | w_t)
```

**Negative Sampling:**
To efficiently compute softmax, Word2Vec uses negative sampling:

```
log σ(v'_{w_O}^T v_{w_I}) + Σ_{i=1}^k E_{w_i ~ P_n}[log σ(-v'_{w_i}^T v_{w_I})]
```

Where:
- σ = sigmoid function
- v_{w_I} = input word vector
- v'_{w_O} = output word vector
- P_n = noise distribution
- k = number of negative samples

**Characteristics:**
- **Advantages**: Captures semantic relationships, smaller dimension than TF-IDF
- **Disadvantages**: Requires training, averages word vectors (loses sequence info)
- **Skip-gram vs CBOW**: Skip-gram better for rare words, CBOW faster to train

**References:**
- Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013)
- arXiv:1301.3781
- https://arxiv.org/abs/1301.3781

### 3. BERT (Bidirectional Encoder Representations from Transformers)

**Mathematical Foundation:**

**Self-Attention Mechanism:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q = Query matrix (XW^Q)
- K = Key matrix (XW^K)
- V = Value matrix (XW^V)
- d_k = dimension of key vectors (scaling factor)
```

**Multi-Head Attention:**

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Pre-training Tasks:**
1. **MLM (Masked Language Model)**: Randomly mask 15% of tokens and predict them
2. **NSP (Next Sentence Prediction)**: Predict if two sentences are consecutive

**Model Architecture:**
- Multiple layers of Transformer encoders
- Bidirectional context (reads left-to-right and right-to-left simultaneously)
- Position embeddings for sequence order

**Characteristics:**
- **Advantages**: Understands context bidirectionally, transfer learning from massive pre-training, state-of-the-art performance
- **Disadvantages**: Computationally expensive, requires more memory
- **DistilBERT**: Lightweight variant (40% smaller, 60% faster, retains 97% performance)

**References:**
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)
- arXiv:1810.04805
- https://arxiv.org/abs/1810.04805

## 🤖 Classification Models

### 1. Logistic Regression

**Mathematical Foundation:**

Binary classification using sigmoid function:

```
Hypothesis: h_θ(x) = σ(θ^T x) = 1 / (1 + e^(-θ^T x))

Cost Function (Binary Cross-Entropy):
J(θ) = -1/m Σ[y*log(h_θ(x)) + (1-y)*log(1-h_θ(x))]

With L2 Regularization:
J(θ) = -1/m Σ[y*log(h_θ(x)) + (1-y)*log(1-h_θ(x))] + λ/(2m) Σθ_j^2

Gradient Descent:
θ := θ - α ∇J(θ)
```

**Characteristics:**
- Simple, fast, interpretable
- Outputs probability estimates
- Works well with high-dimensional sparse features
- L2 regularization prevents overfitting

**Origin:** Based on logistic function (discovered 1838, applied to classification in 1950s-1970s)

### 2. Random Forest

**Mathematical Foundation:**

Ensemble of decision trees using bagging:

**Bootstrap Aggregating (Bagging):**
1. Create B bootstrap samples (random sampling with replacement)
2. Train decision tree on each sample
3. Aggregate predictions by majority voting

**Splitting Criteria:**

**Gini Impurity:**
```
Gini(D) = 1 - Σ p_i^2

Where p_i is proportion of class i in dataset D
```

**Information Gain:**
```
IG(D, A) = Entropy(D) - Σ (|D_v|/|D|) * Entropy(D_v)

Entropy(D) = -Σ p_i * log_2(p_i)
```

**Final Prediction:**
```
ŷ = mode{h_1(x), h_2(x), ..., h_B(x)}
```

**Characteristics:**
- Reduces overfitting through averaging
- Handles non-linear relationships
- Provides feature importance
- Robust to outliers and noise

**Origin:** Leo Breiman, "Random Forests" (2001), Machine Learning 45(1): 5-32

### 3. AdaBoost (Adaptive Boosting)

**Mathematical Foundation:**

Iteratively trains weak learners, focusing on misclassified samples:

**Algorithm:**

```
1. Initialize weights: w_i^(1) = 1/N

2. For t = 1 to T:
   a) Train classifier h_t with weights w^(t)
   
   b) Compute weighted error:
      ε_t = Σ w_i^(t) * I(h_t(x_i) ≠ y_i) / Σ w_i^(t)
   
   c) Compute classifier weight:
      α_t = 0.5 * ln((1 - ε_t) / ε_t)
   
   d) Update sample weights:
      w_i^(t+1) = w_i^(t) * exp(-α_t * y_i * h_t(x_i))
      
   e) Normalize weights:
      w_i^(t+1) = w_i^(t+1) / Σ w_j^(t+1)

3. Final hypothesis:
   H(x) = sign(Σ α_t * h_t(x))
```

**Characteristics:**
- Focuses on hard-to-classify samples
- Combines weak learners into strong learner
- Less prone to overfitting than other boosting methods
- Sensitive to noisy data and outliers

**Origin:** Freund & Schapire, "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting" (1996), Journal of Computer and System Sciences 55(1): 119-139

### 4. LSTM (Long Short-Term Memory)

**Mathematical Foundation:**

LSTM solves the vanishing gradient problem in RNNs through gating mechanisms:

**Gates and Updates:**

```
1. Forget Gate (what to forget from cell state):
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

2. Input Gate (what new information to store):
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)

3. Cell State Update (memory update):
   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

4. Output Gate (what to output):
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t ⊙ tanh(C_t)

Where:
- σ = sigmoid function [0, 1]
- tanh = hyperbolic tangent [-1, 1]
- ⊙ = element-wise multiplication
- h_t = hidden state
- C_t = cell state (long-term memory)
```

**Key Innovation:**
- Cell state (C_t) acts as a "memory highway"
- Gradients can flow through time without vanishing
- Gates control information flow (selective forgetting and updating)

**Characteristics:**
- Handles sequential dependencies
- Solves vanishing gradient problem
- Can capture long-range dependencies
- More parameters than traditional ML models

**Origin:** Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997), Neural Computation 9(8): 1735-1780

## 🔧 Experimental Setup

### Hardware & Software
- **Device**: MacBook M2 chip, 8GB RAM
- **Acceleration**: Apple MPS (Metal Performance Shaders)
- **Python**: 3.8+
- **Key Libraries**: scikit-learn, PyTorch, Transformers, Gensim

### Configuration (M2 Optimized)
- **Dataset samples**: 10,000 (reduced for 8GB RAM)
- **BERT model**: DistilBERT (lightweight)
- **Max sequence length**: 256
- **BERT batch size**: 8
- **Word2Vec dimensions**: 100
- **LSTM hidden units**: 64

### Cross-Validation
- **Method**: Repeated Stratified K-Fold
- **Splits**: 5-fold cross-validation
- **Repeats**: 4 times
- **Total iterations**: 5 × 4 = 20 iterations per model
- **Metric**: Accuracy (mean ± std)

### Hyperparameter Tuning
- **Method**: GridSearchCV
- **Search space**: Defined in `config/config.yaml`
- **Selection**: Best parameters based on CV score

## 📊 Results

### Test Accuracy (%)

| Embedding \ Model | Logistic Regression | Random Forest | AdaBoost | LSTM |
|-------------------|:-------------------:|:-------------:|:--------:|:----:|
| TF-IDF | - | - | - | - |
| Word2Vec (CBOW) | - | - | - | - |
| Word2Vec (Skip-gram) | - | - | - | - |
| DistilBERT | - | - | - | - |

> **Note**: Run experiments using `python experiments/run_experiments.py` to generate results.
> 
> **Result Legend**:
> - **Bold** = Highest accuracy in that column (best model for that embedding)
> - *Italic* = Highest accuracy in that row (best embedding for that model)

## 📈 Analysis & Conclusions

### Expected Findings

**Embedding Comparison:**

1. **BERT/DistilBERT** (Expected Best):
   - Pre-trained on massive corpus (contextual understanding)
   - Bidirectional attention captures nuanced meanings
   - Understands negation, sarcasm better than others
   - **Why superior**: Transfer learning + contextual embeddings

2. **Word2Vec** (Expected Medium):
   - Captures semantic similarity well
   - Skip-gram typically better than CBOW for sentiment
   - **Limitation**: Averages word vectors, loses word order
   - Better than TF-IDF but not context-aware like BERT

3. **TF-IDF** (Expected Baseline):
   - Simple bag-of-words approach
   - Works well with keyword-based sentiment
   - **Limitation**: No semantic understanding, no context
   - Can still perform decently on straightforward reviews

**Model Comparison:**

1. **Logistic Regression**:
   - Fast, simple, interpretable
   - Works well with TF-IDF and BERT embeddings
   - Linear decision boundary

2. **Random Forest**:
   - Handles non-linearity
   - Good with Word2Vec embeddings
   - May overfit on noisy data

3. **AdaBoost**:
   - Focuses on hard samples
   - Can be sensitive to outliers
   - Typically lower performance than RF

4. **LSTM**:
   - Best for sequential data
   - Works well with dense embeddings (Word2Vec, BERT)
   - May not add much value when using pre-aggregated features

**Key Insights:**

- **Best combination likely**: DistilBERT + Logistic Regression (simple on powerful features)
- **Fastest**: TF-IDF + Logistic Regression
- **Most complex**: BERT + LSTM (may overfit on small dataset)

## 🚀 Installation & Usage

### 1. Clone Repository

```bash
git clone https://github.com/b211910027must/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

1. Download the IMDB dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Place `IMDB Dataset.csv` in the `data/raw/` directory

### 4. Run Experiments

```bash
python experiments/run_experiments.py
```

This will:
- Load and preprocess the data
- Generate embeddings (TF-IDF, Word2Vec, BERT)
- Train all models with 20-fold cross-validation
- Evaluate on test set
- Save results to `results/` directory

### 5. View Results

Results are saved in:
- `results/results_TIMESTAMP.csv` - CSV format
- `results/results_TIMESTAMP.json` - JSON format

## 📁 Project Structure

```
imdb-sentiment-analysis/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config/
│   └── config.yaml             # Configuration (M2 optimized)
├── data/
│   ├── raw/                    # Place IMDB Dataset.csv here
│   └── processed/              # Processed data (auto-generated)
├── src/
│   ├── utils/
│   │   └── device_utils.py     # M2 MPS support
│   ├── data/
│   │   └── data_loader.py      # Data loading & preprocessing
│   ├── embeddings/
│   │   ├── tfidf_embedding.py
│   │   ├── word2vec_embedding.py
│   │   └── bert_embedding.py
│   ├── models/
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   ├── adaboost.py
│   │   └── lstm_model.py
│   └── evaluation/
│       └── metrics.py
├── experiments/
│   └── run_experiments.py      # Main experiment runner
├── results/                     # Experiment results
└── reports/                     # Additional reports
```

## 📚 References

### Embedding Methods
1. **TF-IDF**: Salton & Buckley (1988), "Term-weighting approaches in automatic text retrieval"
2. **Word2Vec**: Mikolov et al. (2013), "Efficient Estimation of Word Representations in Vector Space", arXiv:1301.3781
3. **BERT**: Devlin et al. (2018), "BERT: Pre-training of Deep Bidirectional Transformers", arXiv:1810.04805

### Classification Models
1. **Random Forest**: Breiman (2001), "Random Forests", Machine Learning 45(1): 5-32
2. **AdaBoost**: Freund & Schapire (1996), "A Decision-Theoretic Generalization of On-Line Learning"
3. **LSTM**: Hochreiter & Schmidhuber (1997), "Long Short-Term Memory", Neural Computation 9(8)

### Dataset
- Maas et al. (2011), "Learning Word Vectors for Sentiment Analysis", ACL 2011

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Student ID: b211910027must
