# Architecture

## System Overview

Confabulation Scaling is a Python package that predicts the likelihood of LLM hallucination (reference recall errors) by jointly modeling the relationship between document frequency in training corpora and model parameter count. The system follows a modular pipeline: it downloads and indexes reference corpora (arXiv, Wikipedia), estimates topic frequencies, collects oracle verification data, and fits a calibrated sigmoid function to predict hallucination probability across different model scales. This enables practitioners to estimate how reliably a language model of a given size will cite or recall specific topics.

## Module Dependency Diagram

```mermaid
graph TB
    subgraph Input["Input & Data"]
        DD["data_downloader.py"]
    end
    
    subgraph Processing["Processing & Indexing"]
        IB["index_builder.py"]
        Corpus["corpus.py"]
    end
    
    subgraph Oracle["Verification & Ground Truth"]
        OR["oracle.py"]
    end
    
    subgraph Model["Scaling Law Model"]
        SF["sigmoid_fitter.py"]
    end
    
    subgraph Init["Package"]
        Init["__init__.py"]
    end
    
    DD -->|abstracts_path, wiki_path| IB
    IB -->|builds index| Corpus
    Corpus -->|frequency estimates| SF
    OR -->|verification scores| SF
    SF -->|fitted model| Init
    IB -->|persists index| Corpus
    
    style DD fill:#e1f5ff
    style IB fill:#fff3e0
    style Corpus fill:#fff3e0
    style OR fill:#f3e5f5
    style SF fill:#e8f5e9
    style Init fill:#fce4ec
```

## Module Descriptions

### `src/confabulation_scaling/__init__.py`

**Role:** Package initialization and public API surface.

Exports principal classes and functions for end-user access: `CorpusFrequencyEstimator`, `DataDownloader`, `IndexBuilder`, `Oracle`, and `SigmoidFitter`. Defines package version and metadata. Serves as the single entry point for downstream users importing the package.

---

### `src/confabulation_scaling/data_downloader.py`

**Role:** Remote data acquisition and caching.

**Class:** `DataDownloader`

**Methods:**
- `fetch_arxiv_abstracts(n: int = 50000) -> Path` — Downloads arXiv paper abstracts via public API and caches them locally as a file (JSON or CSV). Returns filesystem path to downloaded data.
- `fetch_wikipedia_sample(n: int = 10000) -> Path` — Downloads a stratified sample of Wikipedia articles and caches locally. Returns filesystem path to downloaded data.

Handles network requests, retry logic, and local persistence to avoid redundant downloads. Provides reproducible data snapshots for index building.

---

### `src/confabulation_scaling/index_builder.py`

**Role:** Corpus parsing, tokenization, and inverted index construction.

**Class:** `IndexBuilder`

**Methods:**
- `build(abstracts_path: Path, wiki_path: Path) -> dict` — Parses arXiv abstracts and Wikipedia articles from provided file paths, tokenizes/lemmatizes using spaCy, and constructs an inverted index mapping topics/entities to document frequencies.
- `save(index: dict, path: Path) -> None` — Serializes the built index to disk (JSON or pickle).
- `load(path: Path) -> dict` — Deserializes and loads a previously saved index from disk.

The index is the core lookup table consumed by `CorpusFrequencyEstimator` to estimate how frequently a topic appears in reference corpora.

---

### `src/confabulation_scaling/corpus.py`

**Role:** Topic frequency estimation from indexed corpora.

**Class:** `CorpusFrequencyEstimator`

**Methods:**
- `estimate(topic: str) -> float` — Queries the loaded corpus index and returns the normalized frequency (0–1 scale or log frequency) of the given topic. Implements string similarity matching (Levenshtein distance via `python-Levenshtein`) to handle spelling variants and aliases.

Acts as a bridge between the static index and the scaling model, converting raw topic mentions into a normalized frequency signal.

---

### `src/confabulation_scaling/oracle.py`

**Role:** Ground-truth verification and reference accuracy measurement.

**Class:** `Oracle`

**Methods:**
- `verify(reference: dict) -> float` — Accepts a reference dictionary (containing topic, model output, and expected citation) and returns a scalar accuracy score (0–1, where 1 is correct). May integrate with external fact-checking APIs or human annotation frameworks.

Produces labeled data (topic frequency, model parameter count, oracle accuracy score) for training the sigmoid fitter. Decouples verification logic from model training.

---

### `src/confabulation_scaling/sigmoid_fitter.py`

**Role:** Scaling law model fitting and prediction.

**Class:** `SigmoidFitter`

**Methods:**
- `fit(records: list[dict]) -> dict` — Accepts a list of verification records, each containing `{param_count, doc_freq_raw, oracle_score}`. Fits a 2D calibrated sigmoid function: `oracle_score ≈ sigmoid(a * log(param_count) + b * log(doc_freq_raw) + c)` using scipy optimization. Returns fitted parameters as a dictionary.
- `predict(param_count: float, doc_freq_raw: float) -> float` — Uses fitted parameters to predict hallucination probability (0–1) for a given model size and topic frequency.
- `predict_with_ci(param_count: float, doc_freq_raw: float) -> dict` — Returns prediction alongside confidence intervals (lower, upper bounds) estimated via bootstrap resampling or delta method.

Encapsulates the core scaling law model and inference logic.

---

### `tests/__init__.py`

**Role:** Test package marker.

Empty init file marking the `tests/` directory as a Python package for pytest discovery.

---

### `tests/conftest.py`

**Role:** Pytest fixtures and shared test utilities.

Defines reusable fixtures:
- Mock corpus indices
- Sample verification records
- Temporary filesystem paths for I/O testing
- Pre-fitted sigmoid models for deterministic prediction testing

Centralizes test setup to reduce duplication across test files.

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Data Acquisition                                       │
│                                                                   │
│  DataDownloader.fetch_arxiv_abstracts()  →  arxiv_abstracts.json│
│  DataDownloader.fetch_wikipedia_sample() →  wiki_sample.json    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Index Construction                                      │
│                                                                   │
│  IndexBuilder.build(                                             │
│    abstracts_path, wiki_path                                    │
│  ) → { "topic_1": freq, "topic_2": freq, ... }                  │
│                                                                   │
│  IndexBuilder.save(index) → index.pkl                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Frequency Estimation                                    │
│                                                                   │
│  CorpusFrequencyEstimator.estimate("neural networks")            │
│    → 0.87 (normalized frequency)                                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: Oracle Verification (Parallel Data Collection)          │
│                                                                   │
│  For each (model_param_count, topic):                           │
│    - Query LLM with topic                                       │
│    - Oracle.verify