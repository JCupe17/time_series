import os
import pandas as pd


def load_dataset(folder: str, filename: str, dates_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(folder, filename), compression="gzip", parse_dates=dates_columns)
    return df
