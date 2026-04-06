# NLP Embeddings & Text Clustering Analysis

An end-to-end Natural Language Processing (NLP) project focused on transforming text data into word embeddings, applying advanced clustering techniques, and evaluating their performance using dimensionality reduction.

## Project Overview
This project explores how different text vectorization methods and clustering algorithms interact with high-dimensional text data. The goal is to cluster articles and evaluate the semantic cohesion and label matching of the resulting clusters.

### Datasets
The analysis is performed on two standard NLP text datasets:
* **BBC News Dataset:** Contains text data grouped into 5 categories (Business, Entertainment, Politics, Sport, and Tech).
* **20NewsGroups Dataset:** Contains text data grouped into 20 distinct topics.

### Methodologies & Tech Stack
The pipeline consists of extensive Exploratory Data Analysis (EDA), text preprocessing, feature extraction, and machine learning modeling:

**1. Text Preprocessing:**
* Proactive HTML tag removal (via `BeautifulSoup`), URL stripping, and emoticon/chat word translation.
* Removal of standard and domain-specific stopwords, followed by lemmatization using `WordNetLemmatizer`.
* N-gram exploration and WordCloud generation.
    
**2. Feature Extraction (Embeddings):**
* TF-IDF vectorization method
* Word2Vec vectorization method
* FastText vectorization method
   
**3. Clustering Algorithms:**
* K-Means
* Agglomerative
* HDBSCAN
   
**4. Dimensionality Reduction:**
* PCA
* TruncatedSVD (Used specifically as the sparse alternative to PCA for TF-IDF)
    
**5. Evaluation Metrics:**
* Silhouette Score (for internal cluster cohesion)
* NMI (Normalized Mutual Information), ARI (Adjusted Rand Index), and AMI (Adjusted Mutual Information) for external label matching.
    
## Key Results & Insights 

**20NewsGroups Dataset Results:**
* **Winning Overall Model:** The combination of TF-IDF with TruncatedSVD & HDBSCAN yielded the best results (NMI=0.6153, ARI=0.3072, AMI=0.4690 with `min_cluster_size=5`).
* **K-Means:** TF-IDF without PCA (k=18) was the best configuration for K-Means (NMI=0.302, ARI=0.104, AMI=0.277).
* **Agglomerative:** TF-IDF with TruncatedSVD was the best setup for this algorithm (NMI=0.2815, ARI=0.1303, AMI=0.2502).

**BBC News Dataset Results:**
* **Winning Overall Model:** The combination of TF-IDF with PCA & Agglomerative Clustering (k=5) yielded the best overall label matching results (ARI=0.737, NMI=0.712, AMI=0.711).
* **K-Means:** TF-IDF without PCA (k=5) was the best configuration for K-Means. Over-clustering (e.g., k=8) fragmented true topic boundaries, and PCA introduced information loss that degraded quality.
* **HDBSCAN:** TF-IDF with PCA (`min_cluster_size=15`) achieved the best scores for this algorithm (NMI=0.720, ARI=0.534, AMI=0.713), but it found 10 clusters instead of the ground truth 5. Configurations that forced exactly 5 clusters scored lower (ARI=0.451), showing that matching the ground truth cluster count does not guarantee better label alignment for density-based algorithms.

## Tools used in this project:

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-Learn**
* **Gensim**
* **HDBSCAN**
* **NLTK**
* **SpaCy**
* **BeautifulSoup**
* **SciPy**
* **pyLDAvis**

## Clustering Visualizations:

<b>BBC News Clustering:</b><br/>
<img width="602" height="366" alt="BBC_results" src="https://github.com/user-attachments/assets/1563c76a-9085-4a6b-a058-66e83bbdfe17" />
<br /><br />

<b>20 News Group Clustering:</b><br/>
<img width="602" height="355" alt="20_news_group_results" src="https://github.com/user-attachments/assets/a52e5ee7-8c2a-4547-8d2b-e0d4cb8d1064" />
<br /><br />





