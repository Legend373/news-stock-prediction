import pandas as pd
import re

def top_publishers(df, publisher_col="publisher", top_n=10):
    """
    Returns the top N publishers by article count.
    """
    return df[publisher_col].value_counts().head(top_n)


def publisher_article_distribution(df, publisher_col="publisher"):
    """
    Returns article counts per publisher (all publishers).
    """
    return df[publisher_col].value_counts()


def extract_email_domains(df, publisher_col="publisher"):
    """
    Extracts domains from email-style publisher names.
    Returns a series of domain counts.
    """
    # Extract domain after @
    df["domain"] = df[publisher_col].astype(str).str.extract(r'@(.+)$')
    domain_counts = df["domain"].value_counts()
    return domain_counts


def publisher_news_type(df, publisher_col="publisher", category_col="headline"):
    """
    Optional: Group by publisher and analyze the type of news they report
    (e.g., keywords, topics, or categories). Here we count keywords occurrences.
    """
    return df.groupby(publisher_col)[category_col].apply(lambda x: " | ".join(x)).reset_index()
