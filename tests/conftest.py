from mlflow.tracking import MlflowClient
from dynaflow.cli import _deploy, _destroy
import pytest


@pytest.fixture(scope="session", autouse=True)
def mlflow_client():
    """Provisions the infrastructure for testing and tears it down after the testing session."""
    _deploy(
        base_name="test",
        region="eu-central-1",
        read_capacity_units=1,
        write_capacity_units=1,
    )
    mlflow_client = MlflowClient(
        tracking_uri="dynamodb:eu-central-1:test-tracking-store:test-model-registry",
    )
    yield mlflow_client
    _destroy(base_name="test", region="eu-central-1")
