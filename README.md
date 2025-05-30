# Financial News Topic Modeling

Uncover hidden themes in financial news with advanced topic modeling! This project leverages BERTopic, a cutting-edge technique, to analyze financial news articles, offering powerful text preprocessing, vectorization, and insightful topic modeling capabilities.

## 🌟 Key Features

### Advanced Text Preprocessing
- Comprehensive cleaning: Removes special characters, converts to lowercase
- Filters stop words, applies stemming and lemmatization
- Detailed word count analysis for deeper insights

### Flexible Text Vectorization
- Utilizes TF-IDF for robust feature extraction
- Supports N-gram analysis for capturing contextual patterns
- Fully customizable vectorization parameters to suit your needs

### State-of-the-Art Topic Modeling
- Powered by BERTopic, delivering precise and dynamic topic extraction
- Automatically determines optimal topic numbers
- Analyzes topic similarity and document-topic distributions for comprehensive insights

### In-Depth Model Evaluation
- Measures topic diversity and significance with clear metrics
- Provides detailed document distribution statistics
- Generates intuitive visualizations to explore results visually

## 🚀 Getting Started

### Installation

1. Clone the Repository
```bash
git clone https://github.com/Jackksonns/financial-news-topic-modeling.git
cd financial-news-topic-modeling
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

### Usage

1. Prepare Your Data
   - Place your financial news dataset (CSV format) in the `financial news dataset` folder
   - Ensure the CSV has a `Content` column with raw news articles

2. Run the Analysis
```bash
python financial_news_bertopic_analyzer.py
```

3. Explore Results
   - Processed Data: Saved as `processed_data.csv`
   - Evaluation Metrics: Saved as `topic_modeling_evaluation.json`
   - Visualizations: Automatically generated for topic and distribution analysis

## 📊 Data Format

### Dataset Description
This project uses the High-Quality Financial News Dataset, which contains comprehensive financial information including:
- Company announcements (e.g., dividend decisions)
- Agreement updates (e.g., memorandum of understanding extensions)
- Contract signings (e.g., engineering contracts)

The dataset consists of 1,839 data points, each containing 7 fields (columns), providing a substantial foundation for robust analysis.

> **Dataset Citation**:  
> Abualigah, S. M. (2024). High-Quality Financial News Dataset for NLP Tasks. Kaggle. Available at: [https://www.kaggle.com/datasets/sayelabualigah/high-quality-financial-news-dataset-for-nlp-tasks](https://www.kaggle.com/datasets/sayelabualigah/high-quality-financial-news-dataset-for-nlp-tasks)

### Input
- Required: A CSV file with a `Content` column containing raw financial news articles
- Optional: Additional metadata columns (e.g., date, source)

### Output
1. `processed_data.csv`: Enhanced dataset with:
   - Cleaned, stop-word-free, stemmed, and lemmatized text
   - Word counts, topic assignments, and probabilities

2. `topic_modeling_evaluation.json`: Detailed metrics:
   - Topic size distribution and similarity matrix
   - Average topic probability, diversity, and significance scores

3. Visualizations:
   - Topic visualization
   - Topic distribution charts

## 📋 Requirements
Check `requirements.txt` for a complete list of dependencies.

## 🤝 Contributing
Contributions are welcome! Feel free to submit issues, suggest enhancements, or fork the repo to collaborate. As a fellow learner in AI (like myself, @Jackksonns), I'd love to grow this project with the community.

## 📜 License
This project is licensed under the MIT License—see the LICENSE file for details.
