# Confabulation Scaling

<div class="hero-container" markdown>

## 🧠 Predict LLM Factual Recall Errors

**Confabulation Scaling** is a Python package that models how large language models systematically fail to recall facts. Using calibrated sigmoid scaling laws, we jointly model topic frequency and parameter count to predict when and why LLMs hallucinate.

[Get Started](#quick-start){ .md-button .md-button--primary .md-button--lg }
[View on GitHub](#){ .md-button .md-button--lg }

</div>

## Why Confabulation Scaling?

Large language models are powerful but unreliable for factual recall. They "confabulate"—confidently stating false information. Understanding *when* and *how much* they fail is critical for safe deployment.

**Confabulation Scaling** provides:
- **Predictive scaling laws** that forecast recall errors across model sizes
- **Topic frequency modeling** linking corpus statistics to hallucination rates
- **Calibrated uncertainty** via sigmoid-based probability estimates
- **Production-ready tooling** for auditing LLM factuality

---

## Key Capabilities

<div class="grid" markdown>

=== "📊 Corpus Analysis"

    **Estimate topic frequency** across reference corpora using multiple data sources. The `CorpusFrequencyEstimator` measures how often a topic appears in Wikipedia, web text, and other references—the foundation for predicting recall.

=== "📈 Scaling Law Prediction"

    **Model error rates across parameter counts** with calibrated sigmoid functions. Predict how a 7B model will hallucinate differently than a 70B model for the same topic.

=== "🎯 Factuality Auditing"

    **Benchmark LLM recall performance** against predicted baselines. Identify high-risk topics and model sizes where hallucination risk is highest.

=== "⚙️ Production Ready"

    **Modular, tested architecture** with `scipy`, `numpy`, and `spacy` for robust numerical computing and NLP pipelines. Full test coverage and type hints.

</div>

---

## Quick Start

### Installation

```bash
pip install confabulation-scaling
```

Or install from source:

```bash
git clone https://github.com/yourusername/confabulation-scaling.git
cd confabulation-scaling
pip install -e .
```

### Basic Usage

```python
from confabulation_scaling.corpus import CorpusFrequencyEstimator

# Estimate how often a topic appears in reference corpora
estimator = CorpusFrequencyEstimator()
frequency = estimator.estimate("machine learning")

print(f"Topic frequency: {frequency}")
```

---

## Documentation

- **[Getting Started](./getting-started.md)** — Installation, setup, and first predictions
- **[API Reference](./api/index.md)** — Complete module and class documentation
- **[Scaling Laws](./guides/scaling-laws.md)** — Understanding the math behind predictions
- **[Examples](./examples/index.md)** — Tutorials and reproducible notebooks

---

## Architecture

```
src/confabulation_scaling/
├── corpus.py              # Corpus frequency estimation
├── scaling.py             # Scaling law models
├── calibration.py         # Sigmoid calibration
└── audit.py               # LLM factuality auditing
```

**Dependencies:**
- `scipy`, `numpy` — Numerical computing and optimization
- `spacy` — NLP tokenization and entity recognition
- `requests` — HTTP corpus queries
- `python-Levenshtein` — String similarity
- `wikitextparser` — Wikipedia parsing

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./contributing.md) for guidelines.

---

## Citation

If you use **Confabulation Scaling** in your research, please cite:

```bibtex
@software{confabulation_scaling_2024,
  title={Confabulation Scaling: Predicting LLM Factual Recall Errors via Scaling Laws},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/confabulation-scaling}
}
```

---

## License

[MIT License](./license.md)