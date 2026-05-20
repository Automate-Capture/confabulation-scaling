# Confabulation Scaling: Quick Start Guide

Welcome to **confabulation_scaling**—a Python package for predicting LLM reference recall across parameter counts and topic frequencies using calibrated sigmoid scaling laws.

---

## Installation

```bash
pip install confabulation_scaling
```

Or from source:

```bash
git clone <repo>
cd confabulation_scaling
pip install -e .
```

**Dependencies** (auto-installed):
- `scipy`, `numpy`, `requests`, `python-Levenshtein`, `spacy`, `wikitextparser`, `pytest`

---

## Core Concepts

| Component | Purpose |
|-----------|---------|
| **Corpus Frequency** | Estimate how often a topic appears in public datasets |
| **Data Downloader** | Fetch arXiv abstracts and Wikipedia samples for indexing |
| **Index Builder** | Build a frequency index from downloaded corpora |
| **Oracle** | Verify reference accuracy against ground truth |
| **Sigmoid Fitter** | Fit and predict confabulation probability via scaled sigmoid |

---

## Usage Examples

### Example 1: Estimate Topic Frequency from Corpus

```python
from confabulation_scaling.corpus import CorpusFrequencyEstimator

# Initialize estimator
estimator = CorpusFrequencyEstimator()

# Estimate frequency of a topic (0.0–1.0 scale)
freq = estimator.estimate("quantum computing")
print(f"Topic frequency: {freq:.4f}")

# Higher values = more common in training data
freq_rare = estimator.estimate("obscure medieval philosophy")
print(f"Rare topic frequency: {freq_rare:.4f}")
```

---

### Example 2: Build Frequency Index from Downloaded Data

```python
from pathlib import Path
from confabulation_scaling.data_downloader import DataDownloader
from confabulation_scaling.index_builder import IndexBuilder

# Step 1: Download raw corpora
downloader = DataDownloader()
abstracts_path = downloader.fetch_arxiv_abstracts(n=50000)
wiki_path = downloader.fetch_wikipedia_sample(n=10000)

print(f"Downloaded abstracts: {abstracts_path}")
print(f"Downloaded Wikipedia: {wiki_path}")

# Step 2: Build index
builder = IndexBuilder()
index = builder.build(abstracts_path, wiki_path)

# Step 3: Save index for later use
index_save_path = Path("./models/freq_index.pkl")
builder.save(index, index_save_path)

# Step 4: Load index
loaded_index = builder.load(index_save_path)
print(f"Index contains {len(loaded_index)} unique topics")
```

---

### Example 3: Fit Scaling Law & Predict Confabulation Rate

```python
from confabulation_scaling.sigmoid_fitter import SigmoidFitter
from confabulation_scaling.oracle import Oracle

# Prepare training data: list of records with measured confabulation rates
training_records = [
    {
        "param_count": 7e9,       # 7B parameters
        "doc_freq_raw": 0.042,    # topic frequency in corpus
        "confabulation_rate": 0.15 # measured from oracle
    },
    {
        "param_count": 13e9,
        "doc_freq_raw": 0.042,
        "confabulation_rate": 0.08
    },
    {
        "param_count": 70e9,
        "doc_freq_raw": 0.042,
        "confabulation_rate": 0.03
    },
    # ... more records
]

# Fit sigmoid scaling law
fitter = SigmoidFitter()
fit_params = fitter.fit(training_records)
print(f"Fitted parameters: {fit_params}")

# Predict confabulation rate for a new model size
param_count = 35e9  # 35B parameter model
topic_freq = 0.042  # estimated topic frequency

predicted_rate = fitter.predict(param_count, topic_freq)
print(f"Predicted confabulation rate: {predicted_rate:.3f}")

# Get prediction with confidence interval
pred_with_ci = fitter.predict_with_ci(param_count, topic_freq)
print(f"Prediction: {pred_with_ci['mean']:.3f} ± {pred_with_ci['std']:.3f}")

# Verify against oracle (ground truth)
oracle = Oracle()
reference = {
    "text": "The model claimed that...",
    "topic": "quantum computing",
    "param_count": param_count
}
accuracy = oracle.verify(reference)
print(f"Oracle verification: {accuracy:.3f}")
```

---

## Workflow Diagram

```
1. Download Data
   ↓
   DataDownloader.fetch_arxiv_abstracts()
   DataDownloader.fetch_wikipedia_sample()
   ↓
2. Build Index
   ↓
   IndexBuilder.build() → IndexBuilder.save()
   ↓
3. Estimate Frequencies
   ↓
   CorpusFrequencyEstimator.estimate(topic)
   ↓
4. Collect Training Data
   ↓
   Oracle.verify(reference) × N
   ↓
5. Fit Scaling Law
   ↓
   SigmoidFitter.fit(records)
   ↓
6. Make Predictions
   ↓
   SigmoidFitter.predict(param_count, freq)
   SigmoidFitter.predict_with_ci(param_count, freq)
```

---

## Key Parameters

| Parameter | Type | Range | Meaning |
|-----------|------|-------|---------|
| `param_count` | float | 1e6 – 1e12 | Model parameter count |
| `doc_freq_raw` | float | 0.0 – 1.0 | Topic frequency in corpus |
| `confabulation_rate` | float | 0.0 – 1.0 | Measured hallucination probability |

---