# Project Report: Financial News Topic Modeling

**Jackson ZHOU**

------

## Abstract

This project leverages the *High-Quality Financial News Dataset* from Kaggle to perform topic modeling, uncovering key themes in financial news such as corporate governance, financial performance, and regulatory compliance. Using the BERTopic model, 27 distinct topics were identified, with a topic diversity score of 0.7663, indicating robust thematic separation. This report reflects on the methodology, results, and potential improvements, aiming to enhance my data analysis skills as an Information Management and Information System (IMIS) student.

------

## 1. Introduction

For this project, I selected the *High-Quality Financial News Dataset* from Kaggle, which contains 1,839 financial news entries across seven fields, including company announcements, agreement updates, and contract signings. This dataset, stripped of sensitive personal information, adheres to ethical standards and provides a rich source for text analysis. As an IMIS student with an interest in finance-related data, I chose this dataset to sharpen my data analysis skills through topic modeling, aiming to extract meaningful insights from unstructured financial news.

------

## 2. Literature Review

### 2.1 Analyzing News and Social Media Data

Natural Language Processing (NLP) has become a cornerstone for analyzing news and social media data. Kumar et al. (2021) highlight the role of text mining in services management, noting that social media platforms—including news streams—are critical for market and competitive intelligence. Their systematic review emphasizes topic modeling as an effective tool for theme extraction, a principle directly applicable to financial news analysis.

### 2.2 Relevant Techniques and Algorithms

Topic modeling techniques have evolved significantly, with methods like Latent Dirichlet Allocation (LDA), Non-negative Matrix Factorization (NMF), and Biterm Topic Model (BTM) widely used. Palanivinayagam et al. (2023) provide a comprehensive review of machine-learning-based text classification, underscoring its versatility in topic modeling. This supports my choice of BERTopic, a modern approach combining Transformer embeddings and clustering, for its ability to handle complex textual data effectively.

------

## 3. Methodology and Analysis

### 3.1 Pipeline

The analysis followed a structured pipeline:

- **Text Preprocessing**: Removed non-alphabetic characters, converted text to lowercase, and applied stemming and lemmatization to standardize words.
- **Vectorization**: Employed `CountVectorizer` and `TfidfVectorizer` to create frequency matrices, prioritizing words with high informational value across documents.
- **Topic Modeling**: Utilized `BERTopic`, which integrates Transformer embeddings with UMAP dimensionality reduction and HDBSCAN clustering, to identify topics and extract keywords.
- **Evaluation Metrics**: Assessed results using Topic Diversity Score (measuring thematic uniqueness) and Topic Significance Score (evaluating topic relevance).
- **Visualization**: Generated interactive visualizations with `topic_model.visualize_topics()` for overall structure and `visualize_distribution()` for per-document topic distributions.

### 3.2 Result Analysis

`BERTopic` identified 27 topics, including an outlier topic (-1), from the dataset. Key observations include:

- **Corporate Governance**: Topic 0 (333 documents) and Topic 4 (58 documents) feature keywords like "meeting," "board," and "shareholder," reflecting a focus on decision-making processes.
- **Financial Performance and Regulation**: Topic 1 (297 documents) centers on financial results, while Topics 5 (55 documents) and 8 (27 documents) involve regulatory terms like "CMA," "approval," and "royal decree."
- **Correlations**: A similarity score of 0.951 between Topic 0 (Corporate Governance) and Topic 2 (Dividends) suggests a strong link, likely due to dividend decisions requiring board and shareholder approval.
- **Diversity**: A Topic Diversity Score of 0.7663 indicates distinct word distributions across topics, affirming the model’s effectiveness in separating themes.

These findings highlight corporate governance, financial analysis, and regulatory compliance as dominant themes, aligning with current trends in financial news.

------

## 4. Research Proposal

While `BERTopic` provided valuable insights, its general-purpose BERT backbone may miss domain-specific nuances. Future work could:

- **Adopt FinBERT**: Use a finance-tuned model to better capture specialized terminology and semantics, improving topic coherence.
- **Incorporate Metadata**: Integrate features like "date," "topic," and "impact" via feature engineering to enhance predictive accuracy.

------

## 5. Conclusion

This project successfully applied `BERTopic` to uncover 27 themes in financial news, revealing insights into corporate governance, financial performance, and regulatory trends. The high topic diversity score reflects a robust model, though domain-specific enhancements could further refine results. Moving forward, I aim to explore advanced models and metadata integration to deepen my analysis, aligning with my IMIS background and career goals in data-driven finance.

------

## References

- Abualigah, S. M. (2024). *High-Quality Financial News Dataset for NLP Tasks*. Kaggle. Available at: https://www.kaggle.com/datasets/sayelabualigah/high-quality-financial-news-dataset-for-nlp-tasks [Accessed 27 May 2025].
- Fan, Y., Shi, L., & Yuan, L. (2023). Topic modeling methods for short texts: A survey. *Journal of Intelligent and Fuzzy Systems, 45*(2), 1971–1990. Available at: https://content.iospress.com/articles/journal-of-intelligent-and-fuzzy-systems/ifs233551 [Accessed 27 May 2025].
- Kumar, S., Kar, A. K., & Ilavarasan, P. V. (2021). Applications of text mining in services management: A systematic literature review. *International Journal of Information Management: Data Insights, 1*(1), 100008. Available at: https://www.sciencedirect.com/science/article/pii/S266709682100001X [Accessed 27 May 2025].
- Palanivinayagam, A., El-Bayeh, C. Z., & Damaševičius, R. (2023). Twenty Years of Machine-Learning-Based Text Classification: A Systematic Review. *Algorithms, 16*(5), 236. Available at: https://www.mdpi.com/1999-4893/16/5/236 [Accessed 27 May 2025].