# TIME SERIES

This repository contains a basic code for a simple time series statistical model: SARIMA. The idea of this repository is to show a simple pipeline to load data, train a model and serve the predictions.


---

## Repository's tree

```bash
.
├── Dockerfile
├── LICENSE
├── README.md
├── data
│   └── .gitkeep
├── models
│   └── .gitkeep
├── notebooks
│   └── EDA.ipynb
├── requirements.txt
└── src
    ├── __init__.py
    ├── app.py
    ├── predict.py
    ├── preprocessing.py
    ├── training.py
    └── utils.py
```

Before to use any script, please be sure to put the data for the train and test in the folder `data`.

All the generated models should be stored in the folder `models`.

You can find a data exploration on the `notebooks/EDA.ipynb` notebook.

You can find the code for the pipeline in the `src` folder.

---

## Docker image

To serve the predictions, you can use the Dockerfile of this repository to run a Flask API that has 2 endpoints.

To build the docker image, you can use the following command:

```bash
docker build -t time_series:latest .
```

To run the docker image, you can use the following command:

```bash
docker run -id --name time_series_1 -p 5000:5000 time_series:latest
```

Then you can access the 2 endpoints in a webbrowser:

* An endpoint that serves the results for the test dataset:
    
    [http://0.0.0.0:5000/query-results?dpt_num=88&but_num=1](http://0.0.0.0:5000/query-results?dpt_num=88&but_num=1)

    This endpoint has 2 different arguments: `dpt_num` and `but_num`.

* An endpoint that serves the results computed by the model during the API call:

    [http://0.0.0.0:5000/query-results?dpt_num=88&but_num=1&nb_weeks=1](http://0.0.0.0:5000/query-results?dpt_num=88&but_num=1&nb_weeks=1)

    This endpoint has 3 different arguments: `dpt_num`, `but_num` and `nb_weeks`.

---

## Next steps in Modeling

Since this repository presents a simple ML algorithm for forecasting without any hyper-parameter optimization. Here are some ideas to improve the model:

* Optimize the hyper-parameters for the SARIMA models.
* Use another algorithm or another approach to compare different models. For instance, using pycaret could be a good idea to compare different models in a esay way.
* A limitation with SARIMA is that it uses only one feature. Other approaches could use more than one feature. For this, we can use other approaches using deep learning models such as LSTM, DeepAR, etc.
* For new stores or stores that do not have a history, we can use the information for similar stores for the forecasting to avoid cold start.
* We can use **MLFlow** to monitor and save models to track their performance.

## Next steps in ML Pipeline

We can do different steps in the pipeline to load, train and serve in the ML pipeline. For example, we can use **airflow** to run the pipeline.

* The load step could imply loading data to a database.
* The training step could be programmed to run every week or month.
