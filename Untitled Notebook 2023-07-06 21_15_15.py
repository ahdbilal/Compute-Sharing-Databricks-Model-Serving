# Databricks notebook source
import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints/Multimodel/invocations'
  headers = {'Authorization': f'Bearer dapi207b1870396e41f362deec6bb7958cef', 
'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

  return response.json()

# COMMAND ----------

data = {
  "dataframe_split": {
    "columns": [
      "alcohol",
      "malic_acid",
      "ash",
      "alcalinity_of_ash",
      "magnesium",
      "total_phenols",
      "flavanoids",
      "nonflavanoid_phenols",
      "proanthocyanins",
      "color_intensity",
      "hue",
      "od280/od315_of_diluted_wines",
      "proline",
      "model"
    ],
    "data": [
      [
        13.64,
        3.1,
        2.56,
        15.2,
        116.5,
        2.7,
        3.03,
        0.17,
        1.6600000000000001,
        5.1,
        0.96,
        3.36,
        845.2,
        "bilal_wine_model"
      ]
    ]
  }
}

df = pd.DataFrame(data["dataframe_split"]["data"], columns=data["dataframe_split"]["columns"])

# COMMAND ----------

score_model(df)

# COMMAND ----------

data_types = df.dtypes
print(data_types)

# COMMAND ----------


