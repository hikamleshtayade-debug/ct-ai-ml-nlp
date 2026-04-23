# Expense Classification: Automated Expense Classification Pipeline

## Overview
This notebook presents a **complete end-to-end machine learning pipeline** for multi-class text classification. The objective is to predict expense categories ( **Services**, **Equipment**, and **Material** ) from unstructured expense remarks. The solution demonstrates both **traditional ML** (TF-IDF + statistical classifiers) and **modern NLP** (LLM-based zero-shot classification) approaches.

---

## Table of Contents
1. [Architectural Insights](#architectural-insights)
   - [Architectural Strategy](#architectural-strategy)
   - [Key Performance Metrics](#key-performance-metrics)
   - [Alternative: Zero-Shot Inference](#alternative-zero-shot-inference)
2. [Key Findings](#key-findings)
   - [Data Preparation & Auto-Labeling Strategy](#1-data-preparation--auto-labeling-strategy)
   - [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
   - [Machine Learning Performance](#3-machine-learning-performance)
   - [Alternative Approach: LLM-Based Zero-Shot Classification](#4-alternative-approach-llm-based-zero-shot-classification)
   - [Zero-shot Classification - Considerations and Recommendations](#41-zero-shot-classification---considerations-and-recommendations)
   - [Selection Summary](#42-selection-summary)
3. [Recommendations & Deployment Path](#recommendations--deployment-path)
4. [Next Steps](#next-steps)
5. [Conclusion](#conclusion)

---

## Architectural Insights

### Architectural Strategy
* **Hybrid Labeling:** Leveraged unsupervised embedding-based clustering ( *all-MiniLM-L6-v2* ) to bootstrap "silver labels," eliminating the need for immediate manual annotation.
* **Text Processing:** Implemented a standardized cleaning pipeline to normalize noise while preserving domain-specific semantic signals.
* **Model Selection:** Evaluated a dual-path approach: Traditional Statistical ML for high-speed production and Transformer-based Zero-Shot for cold-start flexibility.

### Key Performance Metrics
The pipeline was validated using 5-fold stratified cross-validation to ensure reliability across imbalanced classes.

| Model | Macro F1 | Balanced Accuracy |
| :--- | :---: | :---: |
| **Logistic Regression (L2)** | **0.78** | **0.79** |
| Linear SVC | 0.75 | 0.77 |
| Multinomial Naive Bayes | 0.72 | 0.74 |

> **Primary Insight**: Logistic Regression with class-weight balancing outperformed more complex models, offering the best balance of interpretability and accuracy.

### Alternative: Zero-Shot Inference
For scenarios without historical training data, the pipeline supports a **DistilBART** zero-shot classifier.
* **Efficiency**: Optimized for local inference to ensure zero API costs and data privacy.
* **Throughput**: Supports batch processing (~32 samples/batch) with a configurable confidence threshold (0.55) acknowledging model limits.

---

## Key Findings

### 1. Data Preparation & Auto-Labeling Strategy
- **Dataset**: ~267 expense records with free-text remarks and financial metadata
- **Labeling Approach**: Implemented unsupervised embedding-based clustering using SentenceTransformers (`all-MiniLM-L6-v2`)
  - Encodes remarks into dense vectors, applies KMeans (k=3), manually maps clusters to labels
  - **Advantage**: Eliminates manual annotation burden; provides ground truth without domain expert overhead
  - **Trade-off Analysis**: Notebook evaluates three routes: rule-based, embedding clustering, and LLM-based zero-shot (selected as production-ready)

- **Text Cleaning Pipeline**: Removes numeric-only tokens, special characters; normalizes whitespace; applies lowercase standardization
  - Preserves semantic content while reducing noise

### 2. Exploratory Data Analysis (EDA)
- **Class Imbalance**: Material < Equipment < Services (unbalanced distribution detected)
  - Justifies use of **macro F1-score** and **balanced accuracy** over raw accuracy
- **Financial Distribution**: Debit amounts show right-skewed distribution (log-transformation applied)
  - Services category exhibits higher financial variance
- **Text Characteristics**: Remarks range 5–200+ characters with 1–40 words per sample
  - Keyword analysis reveals domain-specific signals (AC, almirah, pipe, wire, etc.)

### 3. Machine Learning Performance
Evaluated three TF-IDF + classifier pipelines with **5-fold stratified cross-validation**:

| Model | Macro F1 | Balanced Accuracy |
|-------|----------|------------------|
| **Logistic Regression** | **0.78 ± 0.05** | **0.79 ± 0.04** |
| Linear SVC | 0.75 ± 0.06 | 0.77 ± 0.05 |
| Multinomial NB | 0.72 ± 0.07 | 0.74 ± 0.06 |

- **Best Model**: Logistic Regression (balanced L2 regularization, class-weighted)
- **Test Performance**: Macro F1 ≈ 0.78, Balanced Accuracy ≈ 0.79
- **Key Insight**: TF-IDF (unigrams + bigrams) + simple linear model outperforms Naive Bayes; SVC slightly underperforms due to kernel selection

### 4. Alternative Approach: LLM-Based Zero-Shot Classification
The notebook implements a **production-ready alternative** using distilBERT-based zero-shot classifier:
- **Model**: *valhalla/distilbart-mnli-12-3* (2× faster, 1–2% less accurate than BART-large)
- **Advantages**: 
  - Zero API cost (local inference)
  - Offline capability; no data transmission
  - Batch processing support (throughput: ~32 samples/batch)
  - Confidence thresholding (~0.55) guards against low-confidence predictions
- **Trade-off**: Slightly lower precision than fine-tuned TF-IDF models; ideal for cold-start scenarios without labeled data

### 4.1 Zero-shot classification - Considerations and recommendations

#### Zero-Shot Model Comparison Matrix

| Model | Parameters | Best Use Case | Latency | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **BART-Large-MNLI** | ~400M | High-precision classification; Benchmark setting. | High (Slower) | **Highest** (~90%) |
| **DistilBART-12-3** | ~230M | Enterprise production; High throughput. | **Medium (2-3x Faster)** | High (~88%) |
| **FLAN-T5-Small** | ~60M | CPU-only/Edge; Simple keyword-heavy tasks. | **Low (Extremely Fast)** | Moderate |

---

#### Environment-Based Deployment Strategy

| Environment | Recommended Model | Rationale |
| :--- | :--- | :--- |
| **Development (Dev)** | *facebook/bart-large-mnli* | Establish the "Gold Standard" accuracy baseline. Focus on quality over speed during the tuning phase. |
| **UAT (Staging)** | *valhalla/distilbart-mnli-12-3* | Provides a production-like experience. Stakeholders can validate classification logic with realistic response times. |
| **Production (Prod)** | *valhalla/distilbart-mnli-12-3* | **Top Choice**: Best balance of cost-efficiency (VRAM usage) and high performance for scalable pipelines. |
| **Production (Lite)** | *google/flan-t5-small* | Use only if infrastructure is severely limited (e.g., small CPU instances or browser-based processing). |

---

### 4.2 Selection Summary
For **Expense Classification Pipeline**, I will recommend **DistilBART-MNLI-12-3**. It maintains nearly the same semantic understanding as the full BART-Large model while significantly reducing the infrastructure overhead required for 24/7 production availability.


---

## Recommendations & Deployment Path

| Stage | Approach | Rationale |
|-------|----------|-----------|
| **Quick Prototype** | LLM zero-shot | No labeling overhead; bootstraps pipeline |
| **Production (High Accuracy)** | Fine-tuned Logistic Regression | Interpretability, speed, controlled accuracy/latency trade-off |
| **Enterprise Scale** | Ensemble (TF-IDF + zero-shot) | Combines rule-based stability with zero-shot generalization |

---

## Next Steps
1. **Hyperparameter Tuning**: Grid search on TF-IDF params (min_df, max_df, C) to push Macro F1 beyond 0.80
2. **Error Analysis**: Investigate misclassified samples; identify domain-specific linguistic patterns
3. **Active Learning**: Integrate human feedback loop for low-confidence predictions
4. **Continuous Monitoring**: Track prediction drift as new expense categories emerge
5. **Orchestration**: Evaluate an agentic workflow where a supervisor agent routes high-confidence samples to the ML pipeline and low-confidence/novel remarks to the Zero-Shot model or human-in-the-loop.

---

## Conclusion

The pipeline establishes a robust framework for automated expense classification. By achieving a **~0.78 macro F1-score** with a lightweight Logistic Regression model, the solution offers high-throughput performance suitable for enterprise integration. The inclusion of a **Zero-Shot** fallback ensures the system remains resilient to ***cold-start scenarios*** as new expense categories are introduced.
