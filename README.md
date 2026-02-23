# Spam Email Classification Project

Implementation of Naive Bayes and Logistic Regression for text classification.


## Project Overview

This project implements spam email classification using:
1. **Logistic Regression** with L2 regularization (Batch GD, Mini-batch GD, SGD)
2. **Multinomial Naive Bayes** (for Bag of Words representation)
3. **Bernoulli Naive Bayes** (for binary representation)

## Requirements

- Python 3.9 or later
- NumPy (for array operations and basic math)
- NLTK (optional, for stopwords; fallback included)

### Installation

```bash
# Install required packages
pip install numpy nltk --break-system-packages

# Download NLTK stopwords (optional)
python -c "import nltk; nltk.download('stopwords')"
```

## Project Structure

```
.
├── data_preparation.py          # Step 1: Convert emails to CSV
├── logistic_regression.py       # Step 2: Logistic Regression implementation
├── multinomial_naive_bayes.py   # Step 3: Multinomial NB implementation
├── bernoulli_naive_bayes.py     # Step 4: Bernoulli NB implementation
├── run_experiments.py           # Main script to run all experiments
├── README.md                    # This file
├── datasets/                    # Raw email data (not included)
│   ├── enron1/
│   │   ├── train/
│   │   │   ├── spam/
│   │   │   └── ham/
│   │   └── test/
│   │       ├── spam/
│   │       └── ham/
│   ├── enron2/
│   └── enron4/
├── processed_data/              # Generated CSV files
└── results/                     # Experiment results
```

## Usage Instructions

### Step 1: Data Preparation

First, convert the raw email datasets into CSV format:

```bash
python data_preparation.py
```

**What it does:**
- Builds vocabulary from training data only
- Creates Bag of Words (BoW) representation (word counts)
- Creates Bernoulli representation (binary presence/absence)
- Generates 12 CSV files (3 datasets × 2 representations × train/test)

**Output files:**
- `enron1_bow_train.csv`, `enron1_bow_test.csv`
- `enron1_bernoulli_train.csv`, `enron1_bernoulli_test.csv`
- `enron2_bow_train.csv`, `enron2_bow_test.csv`
- `enron2_bernoulli_train.csv`, `enron2_bernoulli_test.csv`
- `enron4_bow_train.csv`, `enron4_bow_test.csv`
- `enron4_bernoulli_train.csv`, `enron4_bernoulli_test.csv`

**Configuration:**
Before running, update the paths in `data_preparation.py`:
```python
base_directory = "./datasets"      # Path to your raw email data
output_directory = "./processed_data"  # Where to save CSV files
```

### Step 2-4: Run All Experiments

Run all experiments at once:

```bash
python run_experiments.py
```

**What it does:**
- Trains Logistic Regression with all 3 GD variants (Batch, Mini-batch, SGD)
- Tunes hyperparameter λ using validation set
- Trains Multinomial Naive Bayes on BoW data
- Trains Bernoulli Naive Bayes on Bernoulli data
- Evaluates all models on test sets
- Saves results to text files

**Output:**
- `results/logistic_regression_results.txt`
- `results/naive_bayes_results.txt`

### Individual Model Training

You can also train models individually:

#### Logistic Regression
```python
from logistic_regression import LogisticRegression, load_dataset, compute_metrics

# Load data
X_train, y_train = load_dataset("processed_data/enron1_bow_train.csv")
X_test, y_test = load_dataset("processed_data/enron1_bow_test.csv")

# Train model
model = LogisticRegression(learning_rate=0.01, lambda_reg=1.0, max_iterations=500)
model.fit_batch_gd(X_train, y_train, verbose=True)

# Evaluate
y_pred = model.predict(X_test)
metrics = compute_metrics(y_test, y_pred)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

#### Multinomial Naive Bayes
```python
from multinomial_naive_bayes import MultinomialNaiveBayes, load_dataset, compute_metrics

# Load BoW data
X_train, y_train = load_dataset("processed_data/enron1_bow_train.csv")
X_test, y_test = load_dataset("processed_data/enron1_bow_test.csv")

# Train model
model = MultinomialNaiveBayes(alpha=1.0)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
metrics = compute_metrics(y_test, y_pred)
print(f"F1 Score: {metrics['f1']:.4f}")
```

#### Bernoulli Naive Bayes
```python
from bernoulli_naive_bayes import BernoulliNaiveBayes, load_dataset, compute_metrics

# Load Bernoulli data
X_train, y_train = load_dataset("processed_data/enron1_bernoulli_train.csv")
X_test, y_test = load_dataset("processed_data/enron1_bernoulli_test.csv")

# Train model
model = BernoulliNaiveBayes(alpha=1.0)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
metrics = compute_metrics(y_test, y_pred)
print(f"F1 Score: {metrics['f1']:.4f}")
```

## Implementation Details

### Data Preparation
- **Preprocessing:** Lowercase conversion, punctuation removal, stopword filtering
- **Vocabulary:** Built from training data only (no test set leakage)
- **BoW:** Word frequency counts
- **Bernoulli:** Binary word presence (0 or 1)

### Logistic Regression
- **Algorithm:** Gradient Descent with L2 regularization
- **Variants:** 
  - Batch GD (entire dataset)
  - Mini-batch GD (batch_size=50)
  - Stochastic GD (batch_size=1)
- **Hyperparameters:**
  - Learning rate: 0.01
  - λ values tested: [0.01, 0.1, 1.0, 10.0]
  - Max iterations: 500
  - Validation split: 70% train, 30% validation
- **Loss function:** Cross-entropy + L2 regularization
- **Convergence:** Stops when loss change < 1e-6

### Multinomial Naive Bayes
- **Algorithm:** Based on Stanford NLP textbook (Figure 13.2)
- **Smoothing:** Add-one Laplace smoothing (α=1)
- **Computation:** All calculations in log-space to prevent underflow
- **Data:** Uses Bag of Words representation

### Bernoulli Naive Bayes
- **Algorithm:** Models word presence/absence
- **Smoothing:** Add-one Laplace smoothing (α=1)
- **Computation:** All calculations in log-space
- **Data:** Uses Bernoulli (binary) representation

### Evaluation Metrics
- **Accuracy:** Overall correctness
- **Precision:** Correctness of spam predictions
- **Recall:** Coverage of actual spam emails
- **F1 Score:** Harmonic mean of precision and recall

Positive class = Spam (label=1)

## Expected Runtime

- **Data Preparation:** 1-5 minutes (depending on dataset size)
- **Logistic Regression:** 30-120 seconds per model
- **Naive Bayes:** <5 seconds per model
- **Total Experiment Time:** ~30-60 minutes for all experiments

## Troubleshooting

### Memory Issues
If you encounter memory errors with large vocabularies:
```python
# In data_preparation.py, add minimum word frequency filtering:
word_counts = Counter()
# Count word frequencies first
# Then filter: vocabulary = [w for w in vocab if word_counts[w] >= min_freq]
```

### Slow Convergence
If logistic regression doesn't converge:
- Try different learning rates (0.001, 0.01, 0.1)
- Increase max_iterations
- Check for feature scaling issues (though not required for this project)

### File Not Found
Ensure directory structure is correct:
```bash
# Check that datasets exist
ls datasets/enron1/train/spam/
ls datasets/enron1/train/ham/

# Create output directories
mkdir -p processed_data
mkdir -p results
```

## Citations

- NLTK for text preprocessing (stopwords): Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O'Reilly Media Inc.
- Multinomial Naive Bayes algorithm: http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf
- Dataset: Metsis et al., "Spam Filtering with Naive Bayes - Which Naive Bayes?" CEAS 2006

