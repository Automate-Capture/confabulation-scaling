# Confabulation Scaling API Reference

A Python package for modeling LLM reference recall via predictive scaling laws, jointly accounting for topic frequency and parameter count through calibrated sigmoid functions.

---

## Module Overview

| Module | Purpose |
|--------|---------|
| `corpus.py` | Estimate topic frequency from corpora |
| `data_downloader.py` | Fetch arXiv and Wikipedia data |
| `index_builder.py` | Build and manage reference indices |
| `oracle.py` | Verify reference accuracy |
| `sigmoid_fitter.py` | Fit and predict scaling laws |

---

## `src/confabulation_scaling/corpus.py`

**Purpose:** Estimate the frequency/prevalence of topics in reference corpora.

### `CorpusFrequencyEstimator.estimate(topic: str) -> float`

Estimate the frequency of a given topic in the reference corpus.

**Parameters:**
- `topic` (str): The topic or reference to estimate frequency for.

**Returns:**
- `float`: Frequency estimate (normalized between 0 and 1, or raw count depending on implementation).

**Example:**
```python
from confabulation_scaling.corpus import CorpusFrequencyEstimator

estimator = CorpusFrequencyEstimator()
freq = estimator.estimate("transformer attention mechanism")
print(f"Topic frequency: {freq}")
```

---

## `src/confabulation_scaling/data_downloader.py`

**Purpose:** Download and cache reference datasets (arXiv, Wikipedia) for corpus construction.

### `DataDownloader.fetch_arxiv_abstracts(n: int = 50000) -> Path`

Fetch arXiv paper abstracts from the arXiv API and save locally.

**Parameters:**
- `n` (int, optional): Number of abstracts to fetch. Default: 50000.

**Returns:**
- `Path`: File path to saved abstracts (JSON or similar format).

**Example:**
```python
from confabulation_scaling.data_downloader import DataDownloader

downloader = DataDownloader()
abstracts_path = downloader.fetch_arxiv_abstracts(n=10000)
print(f"Abstracts saved to: {abstracts_path}")
```

### `DataDownloader.fetch_wikipedia_sample(n: int = 10000) -> Path`

Fetch a sample of Wikipedia articles.

**Parameters:**
- `n` (int, optional): Number of Wikipedia articles to fetch. Default: 10000.

**Returns:**
- `Path`: File path to saved Wikipedia articles (JSON or similar format).

**Example:**
```python
from confabulation_scaling.data_downloader import DataDownloader

downloader = DataDownloader()
wiki_path = downloader.fetch_wikipedia_sample(n=5000)
print(f"Wikipedia sample saved to: {wiki_path}")
```

---

## `src/confabulation_scaling/index_builder.py`

**Purpose:** Build, save, and load reference indices from corpus data.

### `IndexBuilder.build(abstracts_path: Path, wiki_path: Path) -> dict`

Construct a reference index from downloaded arXiv and Wikipedia data.

**Parameters:**
- `abstracts_path` (Path): Path to file containing arXiv abstracts.
- `wiki_path` (Path): Path to file containing Wikipedia articles.

**Returns:**
- `dict`: Index dictionary mapping references/topics to metadata (e.g., frequency, occurrence count).

**Example:**
```python
from pathlib import Path
from confabulation_scaling.index_builder import IndexBuilder

builder = IndexBuilder()
index = builder.build(
    abstracts_path=Path("data/arxiv_abstracts.json"),
    wiki_path=Path("data/wikipedia_sample.json")
)
print(f"Index contains {len(index)} references")
```

### `IndexBuilder.save(index: dict, path: Path) -> None`

Persist an index to disk.

**Parameters:**
- `index` (dict): Index dictionary to save.
- `path` (Path): File path where index will be saved.

**Returns:**
- `None`

**Example:**
```python
from pathlib import Path
from confabulation_scaling.index_builder import IndexBuilder

builder = IndexBuilder()
index = builder.build(Path("data/abstracts.json"), Path("data/wiki.json"))
builder.save(index, Path("data/reference_index.pkl"))
```

### `IndexBuilder.load(path: Path) -> dict`

Load a previously saved index from disk.

**Parameters:**
- `path` (Path): File path to the saved index.

**Returns:**
- `dict`: Index dictionary.

**Example:**
```python
from pathlib import Path
from confabulation_scaling.index_builder import IndexBuilder

builder = IndexBuilder()
index = builder.load(Path("data/reference_index.pkl"))
print(f"Loaded index with {len(index)} entries")
```

---

## `src/confabulation_scaling/oracle.py`

**Purpose:** Verify the correctness of model-generated references against ground truth.

### `Oracle.verify(reference: dict) -> float`

Verify a reference and return a correctness score.

**Parameters:**
- `reference` (dict): Reference record to verify, typically containing fields like `text`, `source`, `topic`.

**Returns:**
- `float`: Correctness score, typically 0.0 (incorrect) to 1.0 (correct), or binary {0, 1}.

**Example:**
```python
from confabulation_scaling.oracle import Oracle

oracle = Oracle()
reference = {
    "text": "Transformers were introduced by Vaswani et al. in 2017",
    "source": "arxiv",
    "topic": "transformer"
}
score = oracle.verify(reference)
print(f"Reference correctness: {score}")
```

---

## `src/confabulation_scaling/sigmoid_fitter.py`

**Purpose:** Fit scaling law models to empirical data and make predictions about reference recall.

### `SigmoidFitter.fit(records: list[dict]) -> dict`

Fit a sigmoid scaling law model to observation records.

**Parameters:**
- `records` (list[dict]): List of observation records, each containing fields such as:
  - `param_count` (float): Model parameter count.
  - `doc_freq_raw` (float): Document frequency of the topic.
  - `correct` (float or int): Binary correctness label or score.

**Returns:**
- `dict`: Fitted model parameters (e.g., sigmoid coefficients, intercept, scale).

**Example:**
```python
from confabulation_scaling.sigmoid_fitter import SigmoidFitter

fitter = SigmoidFitter()
records = [
    {"param_count": 1e9, "doc_freq_raw": 100, "correct": 0.8},
    {"param_count": 7e9, "doc_freq_raw": 100, "correct": 0.95},
    {"param_count": 1e9, "doc_freq_raw": 10, "correct": 0.3},
]
model_params = fitter.fit(records)
print(f"Fitted parameters: {model_params}")
```

### `SigmoidFitter.predict(param_count: float, doc_freq_raw: float) -> float`

Predict reference recall probability given model parameters, parameter count, and document frequency.

**Parameters:**
- `param_count` (float): Model parameter count.
- `doc_freq_raw` (float): Document frequency (raw count) of the reference topic.

**Returns:**
- `float`: Predicted recall probability (0.0 to 1.0).

**Example:**
```python
from confabulation_scaling.sigmoid_fitter import SigmoidFitter

fitter = SigmoidFitter()
model_params = fitter.fit(records)  # From previous example
fitter.model_params = model_params

prob = fitter.predict(param_count=7e9, doc_freq_raw=100)
print(f"Predicted recall probability: {prob:.3f}")
```

### `SigmoidFitter.predict_with_ci(param_count: float, doc_freq_raw: float) -> dict