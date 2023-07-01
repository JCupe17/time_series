import pandas as pd
from math import sqrt
from datetime import datetime
from tqdm import tqdm
import pickle
import logging

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

from src.preprocessing import preprocessing


def measure_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return sqrt(mean_squared_error(actual, predicted))


def train_test_split(df: pd.DataFrame, date_val: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df.index < date_val]
    val_df = df[df.index >= date_val]
    return train_df, val_df


def train_sarima(df: pd.DataFrame, model_name: str):
    """Trains a SARIMA model - a statistical model - using only the turnover information.
    It considers that the index of the dataframe contains the dates.
    """
    model = sm.tsa.statespace.SARIMAX(df["turnover"], order=(1, 1, 1), seasonal_order=(1,1,1,12))
    results = model.fit(disp=False)

    # Error computation
    predicted = results.predict(1, len(df))
    error = measure_rmse(df["turnover"], predicted)

    with open(model_name, "wb") as f:
        pickle.dump([model, results], f)

    return results, error


def main():
    # 1. Load datasets
    logging.info("Loading training data ...")
    train_df = pd.read_csv("data/train.csv.gz", compression="gzip", parse_dates=["day_id"])

    # 2. Preprocessing
    logging.info("Preprocessing training data ...")
    train_df = preprocessing(train_df)

    # 3. Trainining
    logging.info("Training data by department-store ...")
    all_errors, final_model = [], {}

    for ts in tqdm(train_df["time_series"].unique()):
        # Train and save the model by Department-Store
        df = train_df[train_df["time_series"] == ts]
        results, error = train_sarima(df, model_name=f"models/{ts}.pkl")

        logging.info(f"Model: {ts} - Training RMSE error: {error}")

        # Store errors and models for future analysis
        all_errors.append(error)
        final_model[ts] = results


if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    main()
    logging.info("TRAINING DONE ..")