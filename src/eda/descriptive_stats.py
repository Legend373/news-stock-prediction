import pandas as pd

def add_headline_length(df, text_column="headline"):
    """
    Adds a new column 'headline_length' measuring character length.
    """
    df["headline_length"] = df[text_column].astype(str).str.len()
    return df


def headline_length_stats(df):
    """
    Returns descriptive statistics for headline length.
    """
    return df["headline_length"].describe()


def count_articles_by_publisher(df, publisher_col="publisher"):
    """
    Returns the number of articles per publisher.
    """
    return df[publisher_col].value_counts()


def prepare_publication_date(df, date_col="date"):
    """
    Converts date column to datetime and extracts useful date components.
    """
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["date_only"] = df[date_col].dt.date
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day

    return df


def publication_trends(df, date_col="date_only"):
    """
    Returns the number of articles published per day.
    """
    return df.groupby(date_col).size()
