import os
import pickle
from flask import Flask, request, jsonify

from src.predict import forecast
from src.utils import load_dataset


app = Flask(__name__)

@app.route("/query-results")
def query_results():
    # if key doesn't exist, returns None
    but_num = request.args.get('but_num')
    dpt_num = request.args.get('dpt_num')

    df = load_dataset(folder="data", filename="pred_test.csv.gz", dates_columns=["day_id"])

    if but_num is not None and dpt_num is not None:
        df = df[(df["but_num_business_unit"] == int(but_num)) &
                (df["dpt_num_department"] == int(dpt_num))]
        return jsonify(df[["day_id", "predicted_turnover"]].to_dict('records'))

    return f"No arguments for a given store-department tuple. Please use the arguments but_num and dpt_num."


@app.route("/query-model")
def query_model():
    # if key doesn't exist, returns None
    but_num = request.args.get('but_num')
    dpt_num = request.args.get('dpt_num')
    nb_weeks = request.args.get('nb_weeks')

    if nb_weeks is None:
        nb_weeks = 1

    if but_num is not None and dpt_num is not None:
        model_path = f"models/dpt_{dpt_num}_bu_{but_num}.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                _, results = pickle.load(f)

            # Prepare the dataset
            pred = forecast(results, int(nb_weeks))
            pred.index.name = "day_id"
            pred.columns = ["predicted_turnover"]
            pred = pred.rename(columns={"predicted_mean": "predicted_turnover"})
            pred = pred.reset_index()

            return jsonify(pred[["day_id", "predicted_turnover"]].to_dict('records'))

        return f"No model available for the but_num {but_num} and department {dpt_num}"

    return f"No arguments for a given store-department tuple. Please use the arguments but_num, dpt_num and nb_weeks."


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, host="0.0.0.0", port=5000)
