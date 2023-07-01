from flask import Flask, request, jsonify

from src.utils import load_dataset


app = Flask(__name__)

@app.route("/query", methods=['POST'])
def query():
    # if key doesn't exist, returns None
    but_num = request.args.get('but_num')
    dpt_num = request.args.get('dpt_num')

    df = load_dataset(folder="data", filename="pred_test.csv.gz", dates_columns=["day_id"])

    if but_num is not None and dpt_num is not None:
        df = df[(df["but_num_business_unit"] == int(but_num)) &
                (df["dpt_num_department"] == int(dpt_num))]
    
        return jsonify(df[["day_id", "predicted_turnover"]].to_dict('records'))

    return f"Test DPT {dpt_num} BU {but_num}"


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)