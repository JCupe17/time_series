import os
import pandas as pd
from tqdm import tqdm
import pickle
import logging

from src.preprocessing import preprocessing


def forecast(model, df: pd.DataFrame) -> pd.Series:
    nb_steps = len(df)
    df["predicted_turnover"] = model.get_forecast(steps=nb_steps).predicted_mean.to_frame()["predicted_mean"]
    return df


def predict():
    # 1. Load dataset
    logging.info("Loading data for prediction ...")
    test_df = pd.read_csv("test_data_scientist/test.csv.gz", compression="gzip", parse_dates=["day_id"])

    # 2. Preprocessing
    logging.info("Preprocessing data for prediction ...")
    test_df = preprocessing(test_df)

    # 3. Load model and forecast
    logging.info("Forecasting by department-store ...")
    pred_test_df = pd.DataFrame()
    for ts in tqdm(test_df["time_series"].unique()):
        # Load the model
        model_path = f"models/{ts}.pkl"
        if os.path.exists(model_path):
            with open(model_path, "r") as f:
                _, results = pickle.load(f)
        
            # Prepare the dataset
            df = test_df[test_df["time_series"] == ts]
            df = forecast(results, df)

            pred_test_df = pd.concat([pred_test_df, df], ignore_index=True)
    
    # 4. Save results
    logging.info("Saving forecasting results ...")
    pred_test_df.to_csv("data/pred_test_df.csv.gz", compression="gzip", index=False)


if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    predict()
    logging.info("PREDICTIONS DONE ..")