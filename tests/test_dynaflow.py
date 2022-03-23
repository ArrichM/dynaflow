from mlflow.tracking import MlflowClient


def test_roundtrip_experiment(mlflow_client: MlflowClient):
    """Creates an experiment, gets it, and deletes it."""

    # Create experiment
    experiment_name = "Test Roundtrip Experiment"
    experiment_id = mlflow_client.create_experiment(experiment_name)
    # Retrieve by ame and id
    experiment_by_id = mlflow_client.get_experiment(experiment_id)
    experiment_by_name = mlflow_client.get_experiment_by_name(experiment_name)

    assert experiment_by_id.name == experiment_name
    assert experiment_by_id.__dict__ == experiment_by_name.__dict__

    # Cleanup
    mlflow_client.delete_experiment(experiment_id)


def test_roundtrip_run(mlflow_client: MlflowClient):
    """Creates an experiment, gets it, and deletes it."""

    # Create experiment
    experiment_name = "Test Roundtrip Run"
    experiment_id = mlflow_client.create_experiment(experiment_name)
    # Create run
    run = mlflow_client.create_run(experiment_id)
    mlflow_client.log_metric(run.info.run_id, "test_metric", 1.0)
    recovered_run = mlflow_client.get_run(run.info.run_id)

    assert recovered_run.data.metrics["test_metric"] == 1.0

    mlflow_client.delete_run(run.info.run_id)
