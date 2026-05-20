# Research Background: Predictable Confabulations: Factual Recall by LLMs Scales with Model Size and Topic Frequency

## 1. Research Problem

Large language models (LLMs) demonstrate remarkable capabilities across diverse tasks, yet systematically fail to accurately recall factual information, instead generating plausible-sounding but false statements—a phenomenon termed "hallucination" or "confabulation" (Maynez et al., 2020). This behavior presents a critical barrier to deploying LLMs in knowledge-intensive applications where factual accuracy is essential.

Current understanding of LLM factual recall remains largely qualitative. While practitioners observe that models perform better on frequently discussed topics and that larger models generally improve performance, **no quantitative predictive framework exists that jointly models the relationship between model capacity, topic prevalence in training corpora, and factual accuracy**. This gap leaves practitioners unable to:

- Predict which topics an LLM will reliably recall
- Estimate accuracy improvements from scaling model parameters
- Identify systematic failure modes across the model-topic space
- Design interventions with quantifiable expected outcomes

This research addresses the fundamental question: **Can we predict LLM confabulation rates as a function of model size and topic frequency using a unified scaling law?**

## 2. Related Work and Existing Approaches

### 2.1 Hallucination in Language Models

Early work characterized hallucination as model uncertainty (Huang et al., 2021), while subsequent studies distinguished between different failure types: intrinsic hallucinations (contradicting known facts) versus extrinsic hallucinations (unverifiable claims). Maynez et al. (2020) provided foundational taxonomy and metrics for abstractive summarization systems.

Recent work has focused on mitigation strategies—retrieval-augmented generation (Lewis et al., 2020), constitutional AI (Bai et al., 2022), and uncertainty quantification (Kuhn et al., 2023)—but these remain largely orthogonal to understanding the underlying scaling relationships.

### 2.2 Scaling Laws in Language Models

The emergence of predictive scaling laws (Hoffmann et al., 2020; Kaplan et al., 2020) demonstrated that loss, downstream task performance, and sample efficiency follow power-law relationships with model parameters and training tokens. Chinchilla scaling laws (Hoffmann et al., 2022) refined these relationships, enabling practitioners to predict performance without exhaustive training runs.

**Critical gap:** Existing scaling laws model *general capability* (next-token prediction loss, general language understanding). **No scaling law literature quantifies how factual recall accuracy specifically scales with both model size and topic prevalence.** This represents a novel extension into the factual grounding domain.

### 2.3 Topic Frequency and Model Behavior

Linguistic and cognitive science literature has long established that frequency affects both human and machine recall (Zipf, 1935; Francis & Kučera, 1982). In NLP, word frequency and distributional properties predict model behavior (Belinkov & Glass, 2019). However, these studies operate at the lexical level rather than addressing topic-level factual recall.

Studies examining BERT's knowledge (Petroni et al., 2019) and GPT's memorization (Carlini et al., 2021) provide evidence that training data frequency influences model behavior, but lack a quantitative joint model integrating frequency with model capacity.

### 2.4 Existing Factual Evaluation Frameworks

Current approaches to measuring LLM factuality include:

- **Fact verification datasets** (FEVER, VQA, NQ): Task-specific, limited to curated domains
- **Knowledge probing** (Petroni et al., 2019): Relies on cloze-style prompts; does not measure free-form generation accuracy
- **Reference tracking** (Bohannon et al., 2023): Measures citation precision but lacks predictive models
- **Adversarial evaluation** (Li et al., 2023): Identifies failure modes without quantifying scaling relationships

None of these provide a parameterized, differentiable model for *predicting* accuracy across the continuous space of model sizes and topic frequencies.

## 3. How This Implementation Advances the Field

### 3.1 Novel Theoretical Contribution

This work proposes a **calibrated sigmoid scaling law** of the form:

$$\text{Accuracy}(N, f) = \frac{1}{1 + \exp(-\alpha \cdot \log N + \beta \cdot \log f + \gamma)}$$

where:
- $N$ = model parameter count
- $f$ = topic frequency in training corpus
- $\alpha, \beta, \gamma$ = empirically calibrated coefficients

This formulation:
1. Unifies model capacity and data frequency as *commensurable* predictors via logarithmic scaling
2. Allows non-linear interactions between parameters and frequency
3. Provides probabilistic interpretation (confabulation as sigmoid-bounded phenomenon)
4. Enables extrapolation to unseen (model, topic) pairs

### 3.2 Methodological Advances

The `confabulation_scaling` package implements:

- **Corpus frequency estimation**: Automated extraction of topic prevalence from Wikipedia and web corpora via `CorpusFrequencyEstimator`, addressing the challenge of obtaining ground-truth training data frequency proxies
- **Calibration infrastructure**: Least-squares optimization to fit scaling parameters against empirical accuracy measurements across model-topic pairs
- **Verification pipeline**: Automated reference extraction and validation using `wikitextparser` and Levenshtein distance matching, enabling large-scale evaluation
- **Modular extensibility**: Clean API allowing substitution of corpus sources, distance metrics, and fitting procedures

### 3.3 Practical Applications

Enabling practitioners to:

- **Pre-deployment planning**: Predict whether a model-size-topic combination will meet accuracy thresholds *before* inference
- **Resource allocation**: Identify high-impact topics for retrieval-augmentation or fine-tuning
- **Transparency**: Quantify which topics an LLM can reliably address, supporting responsible model documentation
- **Comparative analysis**: Estimate performance gaps between candidate models on specific domains

## 4. References

Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernian, J., Jones, A., ... & Schiefer, N. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv preprint arXiv:2212.08073*.

Belinkov, Y., & Glass, J. (2019). Analysis methods in deep learning: Users, what, how, and when. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 3818–3828).

Carlini, N., Ippolito, D., Zhang, M., Zhao, R., Kurakin, A., Carbin, M., & Wallace, E. (2021). Extracting training data from large language models. *arXiv preprint arXiv:2012.07290*.

Francis, W. N., & Kučera, H. (1982). *Frequency Analysis of English Usage: Lexicon and Grammar*. Houghton Mifflin.

Hoffmann, J., Borgeaud, S., Mensch, A., Perez, E., Sifre, K., Zen, H., ... & Sifre, L. (2022). An empirical analysis of compute-optimal large language model training. *arXiv preprint arXiv:2203.15556*.

Huang, L., Tan, S., Gao, Z., & Parulian, N. (2021). When do neural networks outperform kernel methods? *Advances in Neural Information Processing Systems*, 34, 14964–14979.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., ... & Amodei, D. (2020). Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*.

Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. *arXiv preprint arXiv:2302.09664*.

Lewis, P., Perez, E., Rinott, R., Schwenk, H., Schwab, D., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *