# Dynaflow

Dynaflow implements a serverless AWS dynamodb tracking store and model registry for MLFlow. It
allows to directly log runs and models to AWS Dynamodb using your AWS credentials. Further
authorisation can be implemented using Dynamodb fine-grained access control.

## Setup
Dynaflow includes a simple CLI that helps to easily provision the Dynamodb tables. To deploy the
tables, run

```
dynaflow deploy
```

which will deploy two AWS Dynamodb tables. To delete the tables, run

```
dynaflow destroy
```


# Configuration
To use the deployed Dynamodb tables as the backend to your tracking store and model registry,
use a tracking store uri of the following format:

`dynamodb:<region>:<tracking-table-name>:<model-table-name>`

where <tracking-table-name> is the name of the dynamodb table you want to use as tracking backend,
<model-table-name>  is the name of the table used for the model registry and <region> is the region
in which the tables reside.

E.g. when using the python client, you can configure the client to use the dynamodb tracking
backend by running the following statement:

`mlflow.set_tracking_uri("dynamodb:eu-central-1:mlflow-tracking-store:mlflow-model-registry")`

To use a table named "mlflow-tracking-store" for tracking and a table named "mlflow-model-registry" as
the model registry backend. Note that these are also the default names you get when running `dynaflow deploy`.

If you want to log your artifacts to s3 by default, you can set the environment variable `DYNAFLOW_ARTIFACT_BUCKET`:
```
export DYNAFLOW_ARTIFACT_BUCKET=<artifact-bucket-name>
```

When running a tracking server, set the dynamodb tracking backend using the following command:

`
mlflow server
    --backend-store-uri dynamodb:<region>:<tracking-table-name>:<model-table-name>
    --default-artifact-root s3://<artifact-bucket-name>/
``
