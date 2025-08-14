# Finance Topic Explorer

**Uncover key themes in financial news.**

[![GitHub](https://img.shields.io/badge/GitHub-Jackksonns-blue?style=flat-square&logo=github)](https://github.com/Jackksonns)

Uncover meaningful, actionable themes from financial news using BERTopic. This project provides a complete pipeline for preprocessing financial text, vectorizing content, building BERTopic models, evaluating results, and visualizing topic distributions.

---

## Key Features

### Advanced Text Preprocessing

* Comprehensive cleaning: removes special characters and converts text to lowercase
* Stop words filtering, stemming and lemmatization
* Detailed word count analysis and token statistics for deeper insight

### Flexible Text Vectorization

* TF-IDF vectorization for robust feature extraction
* Supports N-gram analysis to capture contextual patterns
* Customizable vectorizer parameters to fit different datasets and goals

### State-of-the-Art Topic Modeling

* Built on BERTopic for dynamic and interpretable topic extraction
* Automatic detection and refinement of topic clusters
* Analysis of topic similarity and document-topic distributions for rich insights

### In-Depth Model Evaluation

* Metrics to measure topic diversity, significance, and coherence
* Detailed document distribution and topic-size statistics
* Automatic generation of visualizations for topic exploration

---

## Getting Started

### Installation

1. Clone the Repository

```bash
git clone https://github.com/Jackksonns/fin-topic-explorer.git
cd fin-topic-explorer
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

> Note: Check `requirements.txt` for exact package versions used in this project.

### Usage

1. Prepare Your Data

   * Place your financial news dataset (CSV format) in the `financial news dataset` folder
   * Ensure the CSV file contains a `Content` column with raw news article text

2. Run the Analysis

```bash
python financial_news_bertopic_analyzer.py
```

3. Explore Results

* **Processed Data:** saved as `processed_data.csv`
* **Evaluation Metrics:** saved as `topic_modeling_evaluation.json`
* **Visualizations:** topic and distribution visualizations are generated automatically

---

## Data Format

### Dataset Description

This project uses the *High-Quality Financial News Dataset*, containing a broad set of financial news types:

* Company announcements (e.g., dividend decisions)
* Agreement updates (e.g., memorandum of understanding extensions)
* Contract signings (e.g., engineering contracts)

The dataset in the original experiments contains 1,839 data points with 7 fields per entry — a solid foundation for topic modeling experiments.

> **Dataset Citation**:
> Abualigah, S. M. (2024). High-Quality Financial News Dataset for NLP Tasks. Kaggle. Available at: [https://www.kaggle.com/datasets/sayelabualigah/high-quality-financial-news-dataset-for-nlp-tasks](https://www.kaggle.com/datasets/sayelabualigah/high-quality-financial-news-dataset-for-nlp-tasks)

### Input

* **Required:** CSV file with a `Content` column containing raw financial news articles
* **Optional:** Additional metadata columns (e.g., `date`, `source`, `company`)

### Output

1. `processed_data.csv` — cleaned and enriched dataset containing:

   * Cleaned, stop-word-free, stemmed and lemmatized text
   * Word counts, token statistics
   * Topic assignments and topic probabilities per document

2. `topic_modeling_evaluation.json` — evaluation metrics including:

   * Topic size distribution and similarity matrix
   * Average topic probability, diversity and significance scores

3. **Visualizations** — generated figures for:

   * Topic maps and topic-term distributions
   * Topic distribution across the corpus

---

## Example Workflow

1. Place dataset CSV in `financial news dataset/` with `Content` column
2. Tune vectorizer parameters in `financial_news_bertopic_analyzer.py` (TF-IDF, n-grams)
3. Run the script to preprocess, fit BERTopic, evaluate and save outputs
4. Inspect `processed_data.csv`, `topic_modeling_evaluation.json`, and the generated visualizations

---

## Requirements

See `requirements.txt` for a complete list of dependencies.
