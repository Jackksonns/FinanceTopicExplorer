import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bertopic import BERTopic
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(str_in):
    str_ex_clean = re.sub("[^A-Za-z']+", " ", str_in).strip().lower()
    return str_ex_clean

def rem_sw(str_in):
    sw = stopwords.words('english')
    tmp_arr = list()
    for word in str_in.split():
        if word not in sw:
            tmp_arr.append(word)
    tmp_arr = ' '.join(tmp_arr)
    return tmp_arr

def token_cnt_sw(str_in, sw_in):
    num_tok = str_in.split()
    if sw_in == "unique":
        num_tok = len(set(num_tok))
    else:
        num_tok = len(num_tok)
    return num_tok

def stem_fun(str_in, sw_in):
    tmp_ar = list()
    if sw_in == "ps":
        ps = PorterStemmer()
    else:
        ps = WordNetLemmatizer()
    for word in str_in.split():
        if sw_in == "ps":
            tmp = ps.stem(word)
        else:
            tmp = ps.lemmatize(word)
        tmp_ar.append(tmp)
    tmp_ar = ' '.join(tmp_ar)
    return tmp_ar

def write_pickle(vec, p_in, sw_in):
    import pickle
    with open(f"{p_in}_{sw_in}.pkl", 'wb') as f:
        pickle.dump(vec, f)

def vec_fun(df_in, lab_in, m_in, n_in, sw_in, p_in):
    if sw_in == "tf":
        vec = CountVectorizer(ngram_range=(m_in, n_in))
    else:
        vec = TfidfVectorizer(ngram_range=(m_in, n_in))
    df_vec = pd.DataFrame(
        vec.fit_transform(df_in).toarray()) #caveat more memory
    df_vec.index = lab_in
    df_vec.columns = vec.get_feature_names_out()
    write_pickle(vec, p_in, sw_in)
    return df_vec, vec

def file_reader(path_in):
    try:
        df = pd.read_csv(path_in)
        return df
    except Exception as e:
        print(f"There an error when reading the file {path_in}: {str(e)}")
        return None

# read csv file
df = file_reader('./financial news dataset/dataset.csv')

# show all the column names of the csv file
print("Column names of the CSV file:")
print(df.columns.tolist())

text_column = 'Content'

# store the original data
df_original = df.copy()

# clean text
df[text_column + '_cleaned'] = df[text_column].apply(clean_text)

# remove stop words
df[text_column + '_no_stopwords'] = df[text_column + '_cleaned'].apply(rem_sw)

# stem extracting
df[text_column + '_stemmed'] = df[text_column + '_no_stopwords'].apply(lambda x: stem_fun(x, "ps"))

# lemmatization
df[text_column + '_lemmatized'] = df[text_column + '_no_stopwords'].apply(lambda x: stem_fun(x, "wn"))

# Wordcounts
df[text_column + '_total_words'] = df[text_column + '_no_stopwords'].apply(lambda x: token_cnt_sw(x, "total"))
df[text_column + '_unique_words'] = df[text_column + '_no_stopwords'].apply(lambda x: token_cnt_sw(x, "unique"))

# Vectorization processing(TF-IDF vectorization)
# use processed text for vectorization
processed_text = df[text_column + '_lemmatized'].fillna('')  # handle empty values
df_vec, vec = vec_fun(processed_text, df.index, 1, 2, "tfidf", "vectorizer")

# Topic Modeling using BERTopic
print("\nPerforming Topic Modeling with BERTopic...")

# Custom domain stopwords
custom_stopwords = ["announces", "company", "date", "inc", "co", "press", "release", "pdf",'non']

# Set vectorizer (term frequency vectorizer)
vectorizer_model = CountVectorizer(stop_words=custom_stopwords)

# Initialize BERTopic with custom vectorizer
topic_model = BERTopic(language="english", 
                      min_topic_size=10,  # minimum size of topics
                      nr_topics="auto",   # automatically determine number of topics
                      calculate_probabilities=True,  # Enable probability calculation
                      vectorizer_model=vectorizer_model)  # Use custom vectorizer with stopwords

# Fit the model
topics, probs = topic_model.fit_transform(processed_text)

# Get topic information
topic_info = topic_model.get_topic_info()
print("\nTopic Information:")
print(topic_info)

# Get top words for each topic
print("\nTop words for each topic:")
for topic in topic_model.get_topics():
    print(f"\nTopic {topic}:")
    print(topic_model.get_topic(topic))

# Add topic information to the dataframe
df['Topic'] = topics
# Fix: Handle probability calculation based on probs dimension
if len(probs.shape) == 1:
    df['Topic_Probability'] = probs
else:
    df['Topic_Probability'] = probs.max(axis=1)

# Model Evaluation
print("\nModel Evaluation Metrics:")

# Topic Size Distribution
topic_sizes = topic_model.get_topic_freq()
print("\nTopic Size Distribution:")
print(topic_sizes)

# Topic Similarity Matrix
try:
    # Calculate topic similarity using cosine similarity between topic embeddings
    topic_embeddings = topic_model.topic_embeddings_
    similarity_matrix = cosine_similarity(topic_embeddings)
    print("\nTopic Similarity Matrix:")
    print(similarity_matrix)
except Exception as e:
    print(f"\nWarning: Could not calculate topic similarity matrix: {str(e)}")

# Document-Topic Distribution Statistics
print("\nDocument-Topic Distribution Statistics:")
print(f"Average topic probability: {df['Topic_Probability'].mean():.4f}")
print(f"Standard deviation of topic probability: {df['Topic_Probability'].std():.4f}")
print(f"Number of documents per topic:")
print(df['Topic'].value_counts())

# Topic Evaluation Metrics
def calculate_topic_metrics(topic_words):
    # Topic Diversity: Measure the uniqueness of words across topics
    all_words = set()
    total_words = 0
    for topic in topic_words:
        topic_words_set = set(word for word, _ in topic[:10])  # Use top 10 words
        all_words.update(topic_words_set)
        total_words += len(topic_words_set)
    diversity_score = len(all_words) / total_words if total_words > 0 else 0
    
    # Topic Significance: Measure the weight distribution of topic words
    significance_scores = []
    for topic in topic_words:
        weights = [weight for _, weight in topic[:10]]
        if weights:
            # Calculate the ratio of top 3 words to total weight
            top_weights = sum(weights[:3])
            total_weight = sum(weights)
            significance = top_weights / total_weight if total_weight > 0 else 0
            significance_scores.append(significance)
    
    avg_significance = np.mean(significance_scores) if significance_scores else 0
    
    return diversity_score, avg_significance

# Calculate topic metrics
topic_words = [topic_model.get_topic(topic) for topic in topic_model.get_topics()]
diversity_score, significance_score = calculate_topic_metrics(topic_words)
print(f"\nTopic Diversity Score: {diversity_score:.4f}")
print(f"Topic Significance Score: {significance_score:.4f}")

# Save evaluation results
evaluation_results = {
    'topic_sizes': topic_sizes.to_dict(),
    'topic_similarity': similarity_matrix.tolist(),
    'avg_topic_probability': float(df['Topic_Probability'].mean()),
    'std_topic_probability': float(df['Topic_Probability'].std()),
    'documents_per_topic': df['Topic'].value_counts().to_dict(),
    'topic_diversity': float(diversity_score),
    'topic_significance': float(significance_score)
}

# Save evaluation results to a file
import json
with open('topic_modeling_evaluation.json', 'w') as f:
    json.dump(evaluation_results, f, indent=4)

# see the results
print("\nProcessing completed! Data preview: ")
print(df.head())
print("\nPreview of the vectorized data (showing only non-zero values):")
# show only non-zero values
print(df_vec[df_vec != 0].fillna(''))

# store the processed data
df.to_csv('processed_data.csv', index=False)

# Visualize topics
try:
    topic_model.visualize_topics()
    topic_model.visualize_distribution(probs[0])
except Exception as e:
    print(f"\nWarning: Could not generate visualization: {str(e)}")

