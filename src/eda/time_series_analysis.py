import pandas as pd

def prepare_datetime(df, date_col="date"):
    """
    Ensure datetime column exists and extract useful components.
    """
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["hour"] = df[date_col].dt.hour
    df["date_only"] = df[date_col].dt.date
    df["weekday"] = df[date_col].dt.day_name()
    return df


def publication_frequency(df, date_col="date_only"):
    """
    Returns a series of number of articles per day.
    """
    return df.groupby(date_col).size()


def detect_spikes(freq_series, threshold_factor=2):
    """
    Detects days where publication frequency is above a certain threshold.
    """
    mean = freq_series.mean()
    std = freq_series.std()
    spikes = freq_series[freq_series > mean + threshold_factor * std]
    return spikes


def hourly_publication_distribution(df, hour_col="hour"):
    """
    Returns number of articles published in each hour of the day.
    """
    return df[hour_col].value_counts().sort_index()


def weekday_publication_distribution(df, weekday_col="weekday"):
    """
    Returns number of articles published per day of the week.
    """
    return df[weekday_col].value_counts().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
