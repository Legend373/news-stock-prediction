import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words("english")


def clean_text(text):
    """
    Clean text by removing punctuation, URLs, numbers, and converting to lowercase.
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(df, text_col="headline"):
    """
    Preprocess headline text and add a new 'clean_headline' column.
    """
    df["clean_headline"] = df[text_col].astype(str).apply(clean_text)
    return df


def extract_keywords(df, text_col="clean_headline", max_features=20):
    """
    Uses CountVectorizer to find the most frequent words in headlines.
    """
    vectorizer = CountVectorizer(stop_words=stop_words, max_features=max_features)
    matrix = vectorizer.fit_transform(df[text_col])
    
    keywords = pd.DataFrame({
        "keyword": vectorizer.get_feature_names_out(),
        "count": matrix.sum(axis=0).A1
    }).sort_values(by="count", ascending=False)
    
    return keywords


def perform_lda_topic_modeling(df, text_col="clean_headline", n_topics=5, n_words=10):
    """
    Performs LDA topic modeling and returns the top words in each topic.
    """
    vectorizer = CountVectorizer(stop_words=stop_words)
    matrix = vectorizer.fit_transform(df[text_col])
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(matrix)

    words = vectorizer.get_feature_names_out()

    topics = {}
    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-n_words:]]
        topics[f"Topic {idx+1}"] = top_words

    return topics
