import os
import pandas as pd
from tqdm import tqdm
import pickle
import logging

from src.preprocessing import preprocessing
from src.utils import load_dataset


def forecast(model, nb_steps: int) -> pd.Series:
    prediction =  model.get_forecast(steps=nb_steps).predicted_mean.to_frame()
    return prediction


def predict():
    # 1. Load dataset
    logging.info("Loading data for prediction ...")
    test_df = load_dataset(folder="data", filename="test.csv.gz", dates_columns=["day_id"])

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
            with open(model_path, "rb") as f:
                _, results = pickle.load(f)
        
            # Prepare the dataset
            df = test_df[test_df["time_series"] == ts]
            pred = forecast(results, len(df))
            df = pd.concat([df, pred], axis=1)
            df = df.reset_index()

            pred_test_df = pd.concat([pred_test_df, df], ignore_index=True)
    
    # 4. Save results
    logging.info("Saving forecasting results ...")
    pred_test_df.to_csv("data/pred_test.csv.gz", compression="gzip", index=False)


if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    predict()
    logging.info("PREDICTIONS DONE ..")