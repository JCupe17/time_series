import pandas as pd


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # Create a unique ID
    df["time_series"] = df.apply(lambda x: f"dpt_{x['dpt_num_department']}_bu_{x['but_num_business_unit']}", axis=1)

    df["month"] = df["day_id"].apply(lambda x: x.month)
    df["year"] = df["day_id"].apply(lambda x: x.year)
    df["day_of_week"] = df["day_id"].apply(lambda x: x.dayofweek)
    df["day_of_year"] = df["day_id"].apply(lambda x: x.dayofyear)

    df = df.set_index("day_id").sort_index()

    return df
