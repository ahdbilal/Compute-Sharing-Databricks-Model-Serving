# Databricks notebook source
# MAGIC %md
# MAGIC ## Define Input Parameters

# COMMAND ----------

models = ['bilal_iris_model', 'bilal_wine_model'] #these models are trained below
multimodel_endpoint_name = 'bilalMultimodel'
workload_size = "small"
scale_to_zero_enabled = True

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Required Dependencies

# COMMAND ----------

import os
import pandas as pd
import tempfile
from packaging.version import parse as parse_version

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, Schema

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import requests

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Two Models for Demonstrating Compute Sharing

# COMMAND ----------

# This cell trains two example models (Iris and Wine) that can be utilized to showcase the concept of compute sharing.
# If you have already logged models, you can skip this step and proceed to the next cells.

# Load the Iris dataset
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target
feature_names_iris = iris.feature_names

# Load the Wine dataset
wine = datasets.load_wine()
X_wine = wine.data
y_wine = wine.target
feature_names_wine = wine.feature_names

# Split both datasets into a training set and a test set
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)

def train_model(dataset_name, X_train, y_train, X_test, y_test, feature_names):
    with mlflow.start_run(run_name=dataset_name):

        # Train the model
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        # Create and log model signature
        input_schema = Schema([ColSpec(DataType.double, name=feature_name) for feature_name in feature_names])
        output_schema = Schema([ColSpec(DataType.integer)])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Log input example
        input_example = pd.DataFrame([X_test[0]], columns=feature_names)

        if (dataset_name == "bilal_iris_model"):
          mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example, pip_requirements=['mlflow==2.4.1'])
        else:
          mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)


        # Register the model
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", dataset_name)

# Train the models
train_model("bilal_iris_model", X_train_iris, y_train_iris, X_test_iris, y_test_iris, feature_names_iris)
train_model("bilal_wine_model", X_train_wine, y_train_wine, X_test_wine, y_test_wine, feature_names_wine)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multi-Model Serving Class for Compute Sharing

# COMMAND ----------

# This code defines a generic class, MultiModelPyfunc, designed to serve multiple MLflow models.

class MultiModelPyfunc(mlflow.pyfunc.PythonModel):
    """
    A class for serving multiple MLflow models.
    """

    def load_context(self, context):
        """
        Load models from the specified directory.
        """
        self.models = {}
        path = context.artifacts['models']
        multimodels = os.listdir(path)
        for model in multimodels:
            self.models[model] = mlflow.pyfunc.load_model(os.path.join(path, model))

    def select_model(self, model_input):
        """
        Select the model based on the 'model' field in the input DataFrame.
        """
        selected_model = str(model_input["model"].iloc[0])
        if selected_model not in self.models.keys():
            available_models = ", ".join(self.models.keys())
            raise ValueError(f"Model '{selected_model}' not found. Available models are: {available_models}")
        return selected_model

    def process_input(self, model_input):
        """
        Remove the 'model' field from the input DataFrame.
        """
        return model_input.drop("model", axis=1)

    def predict(self, context, model_input):
        """
        Make predictions with the selected model.
        """
        selected_model = self.select_model(model_input)
        print(f'Selected model: {selected_model}')
        model = self.models[selected_model]
        processed_input = self.process_input(model_input)
        return model.predict(processed_input)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Function to Merge and Resolve Conflicting Requirements Across Models

# COMMAND ----------

# It is recommended to maintain consistent package versions across all logged models to prevent potential conflicts.
# In cases where differences exist, the following function merges 'requirements.txt' files from each model, prioritizing the highest version number when a conflict arises.

from packaging.version import parse as parse_version, InvalidVersion

def merge_requirements(directory):
    requirements = {}

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.startswith('requirements') and filename.endswith('.txt'):
                with open(os.path.join(root, filename), 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '==' in line:
                            name, version = line.split('==')
                            try:
                                parsed_version = parse_version(version)
                                if name not in requirements or parsed_version > parse_version(requirements.get(name, "0.0")):
                                    requirements[name] = version
                            except InvalidVersion:
                                continue
                        else:
                            requirements[line] = requirements.get(line, "")

    return [f"{name}=={version}" if version else name for name, version in requirements.items()]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the Multi-Model Endpoint

# COMMAND ----------

# Download specified version of each model and store in the temporary directory
artifact_dir = tempfile.mkdtemp()
for name in models:
    model_uri = f'models:/{name}/latest'
    local_path = os.path.join(artifact_dir, name)
    mlflow.artifacts.download_artifacts(model_uri, dst_path=local_path)

# Optional - Combines 'requirements.txt' from logged models, preferring the highest version in version conflicts.
requirements = merge_requirements(artifact_dir)

# Prepare input example to be logged with the model
# model, version = 'bilal_wine_model', '2'
# input_example = pd.read_json(f'{artifact_dir}/{model}/input_example.json', orient='split')
# input_example['model'] = model

# Log the Python model
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="bilal_multimodel",
        python_model=MultiModelPyfunc(),
        artifacts={'models': f'{artifact_dir}/'},
        registered_model_name=multimodel_endpoint_name,
        # input_example=input_example,
        pip_requirements=requirements
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or Update Multimodel Endpoint

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
 
# With the token, you can create our authorization header for our subsequent REST calls
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
  }

# Next you need an endpoint at which to execute your request which you can get from the notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
 
# This object comes from the Java CM - Convert the Java Map opject to a Python dictionary
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)
 
# Lastly, extract the Databricks instance (domain name) from the dictionary
instance = tags["browserHostName"]

from mlflow.tracking.client import MlflowClient
def get_latest_model_version(model_name: str):
  client = MlflowClient()
  models = client.get_latest_versions(model_name, stages=["None"])
  for m in models:
    new_model_version = m.version
  return new_model_version

# COMMAND ----------

my_json = {
  "name": multimodel_endpoint_name,
  "config": {
   "served_models": [{
     "model_name": multimodel_endpoint_name,
     "model_version": get_latest_model_version(multimodel_endpoint_name),
     "workload_size": "Small",
     "scale_to_zero_enabled": True
   }]
 }
}
 
def func_create_endpoint(model_serving_endpoint_name):
  #get endpoint status
  endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
  url = f"{endpoint_url}/{model_serving_endpoint_name}"
  r = requests.get(url, headers=headers)
  if "RESOURCE_DOES_NOT_EXIST" in r.text:  
    print("Creating this new endpoint: ", f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations")
    re = requests.post(endpoint_url, headers=headers, json=my_json)
  else:
    new_model_version = (my_json['config'])['served_models'][0]['model_version']
    print("This endpoint existed previously! We are updating it to a new config with new model version: ", new_model_version)
    # update config
    url = f"{endpoint_url}/{model_serving_endpoint_name}/config"
    re = requests.put(url, headers=headers, json=my_json['config'])
  assert re.status_code == 200, f"Expected an HTTP 200 response, received {re.status_code}"

# COMMAND ----------

func_create_endpoint(multimodel_endpoint_name)

# COMMAND ----------


