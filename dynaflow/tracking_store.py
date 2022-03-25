import logging
from typing import Any, List, Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import os
import uuid

import mlflow
from mlflow.entities.run_status import RunStatus
from mlflow.store.entities import PagedList
from mlflow.store.tracking import (
    DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH,
    SEARCH_MAX_RESULTS_DEFAULT,
)
from mlflow.store.tracking.abstract_store import AbstractStore
from pynamodb.attributes import (
    DiscriminatorAttribute,
    ListAttribute,
    MapAttribute,
    NumberAttribute,
    UnicodeAttribute,
)
from pynamodb.indexes import AllProjection, GlobalSecondaryIndex
from pynamodb.models import Model

log = logging.getLogger(__name__)

_TRACKING_DIR_ENV_VAR = "MLFLOW_TRACKING_DIR"
_TRACKING_BUCKET_ENV_VAR = "DYNAFLOW_ARTIFACT_BUCKET"


def _default_root_dir() -> str:
    if os.environ.get(_TRACKING_BUCKET_ENV_VAR, False):
        s3_bucket = os.environ[_TRACKING_BUCKET_ENV_VAR]
        if not s3_bucket.startswith("s3://"):
            s3_bucket = "s3://" + s3_bucket
        return s3_bucket
    elif os.environ.get(_TRACKING_DIR_ENV_VAR, False):
        return os.environ[_TRACKING_DIR_ENV_VAR]
    else:
        return os.path.abspath(DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH)


class ExperimentNameIndex(GlobalSecondaryIndex):
    class Meta:
        index_name = "mlflow-tracking-store-name-index"
        region = "eu-central-1"
        projection = AllProjection()
        read_capacity_units = 2
        write_capacity_units = 1

    name: str = UnicodeAttribute(hash_key=True)
    lifecycle_stage: str = UnicodeAttribute(range_key=True)


class BaseEntry(Model):
    class Meta:
        table_name = "mlflow-tracking-store"
        region = "eu-central-1"

    cls: str = DiscriminatorAttribute()
    id: str = UnicodeAttribute(hash_key=True)
    name: str = UnicodeAttribute(null=True)
    lifecycle_stage: str = UnicodeAttribute(default="active")
    tags: MapAttribute = MapAttribute(default={})

    name_index = ExperimentNameIndex()


class Experiment(BaseEntry, discriminator="Experiment"):
    """
    DynamoDB abstraction for mlflow Experiments
    """

    artifact_location: str = UnicodeAttribute(null=True)

    def to_mlflow(self) -> mlflow.entities.Experiment:
        return mlflow.entities.Experiment(
            experiment_id=self.id,
            name=self.name,
            artifact_location=self.artifact_location,
            lifecycle_stage=self.lifecycle_stage,
            tags=[
                mlflow.entities.ExperimentTag(key=key, value=value)
                for key, value in self.tags.as_dict().items()
            ],
        )


class Metric(MapAttribute):
    key: str = UnicodeAttribute()
    value: float = NumberAttribute()
    timestamp: float = NumberAttribute()
    step: int = NumberAttribute()

    def to_mlflow(self) -> mlflow.entities.Metric:
        return mlflow.entities.Metric(
            key=self.key, value=self.value, timestamp=self.timestamp, step=self.step
        )


class Run(BaseEntry, discriminator="Run"):
    """
    DynamoDB abstraction for mlflow Runs
    """

    experiment_id: str = UnicodeAttribute()
    user_id: str = UnicodeAttribute(null=True)
    status: int = NumberAttribute(default=2)
    start_time: float = NumberAttribute(null=True)
    end_time: float = NumberAttribute(null=True)
    artifact_uri: str = UnicodeAttribute()

    metrics: List[Metric] = ListAttribute(default=[], of=Metric)
    params: MapAttribute = MapAttribute(default={})

    def to_mlflow(self) -> mlflow.entities.Run:
        run_info = mlflow.entities.RunInfo(
            run_id=self.id,
            run_uuid=self.id,
            experiment_id=self.experiment_id,
            user_id=self.user_id,
            status=RunStatus.to_string(self.status),
            start_time=self.start_time,
            end_time=self.end_time,
            lifecycle_stage=self.lifecycle_stage,
            artifact_uri=self.artifact_uri,
        )
        tags = [
            mlflow.entities.RunTag(key=key, value=value)
            for key, value in self.tags.as_dict().items()
        ]
        params = [
            mlflow.entities.Param(key=key, value=value)
            for key, value in self.params.as_dict().items()
        ]
        metrics = [metric.to_mlflow() for metric in self.metrics]
        run_data = mlflow.entities.RunData(metrics=metrics, params=params, tags=tags)
        return mlflow.entities.Run(run_info, run_data)


class DynamodbTrackingStore(AbstractStore):
    """
    MLFlow tracking store backed by AWS Dynamodb
    """

    def __init__(self, store_uri: str, artifact_uri: str):
        super(DynamodbTrackingStore, self).__init__()
        _, region, tracking_table_name, model_table_name = store_uri.split(":")

        BaseEntry.Meta.table_name = tracking_table_name
        BaseEntry.Meta.region = region
        self.artifact_location = artifact_uri or _default_root_dir()

        if not BaseEntry.exists():
            log.error(
                "Tracking Store table does not exist."
                " Run 'dynaflow deploy' to deploy the necessary tables."
            )

    def list_experiments(
        self,
        view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=None,
        page_token=None,
    ) -> PagedList[mlflow.entities.Experiment]:
        """
        :param view_type: Qualify requested type of experiments.
        :param max_results: If passed, specifies the maximum number of experiments desired. If not
                            passed, all experiments will be returned. However, certain server
                            backend may apply its own limit. Check returned ``PagedList`` token to
                            see if additional experiments are available.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``list_experiments`` call.
        :return: A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
                 :py:class:`Experiment <mlflow.entities.Experiment>` objects. The pagination token
                 for the next page can be obtained via the ``token`` attribute of the object.
        """
        limit = max_results or 100
        if view_type == mlflow.entities.ViewType.ACTIVE_ONLY:
            experiments = Experiment.scan(
                Experiment.lifecycle_stage == "active",
                limit=limit,
                last_evaluated_key=page_token,
            )
        elif view_type == mlflow.entities.ViewType.DELETED_ONLY:
            experiments = Experiment.scan(
                Experiment.lifecycle_stage == "deleted",
                limit=limit,
                last_evaluated_key=page_token,
            )
        else:
            experiments = Experiment.scan(limit=limit, last_evaluated_key=page_token)
        experiments = [experiment for experiment in experiments]

        if len(experiments) == 0:
            return PagedList([], None)

        page_token = experiments[-1].id

        return PagedList(
            [experiment.to_mlflow() for experiment in experiments],
            token=page_token,
        )

    def create_experiment(
        self,
        name: str,
        artifact_location: Optional[str] = None,
        tags: List[mlflow.entities.ExperimentTag] = None,
    ) -> Optional[str]:
        """
        Create a new experiment.
        If an experiment with the given registered_model_name already exists, throws exception.

        :param name: Desired name for an experiment
        :param artifact_location: Base location for artifacts in runs. May be None.
        :param tags: Experiment tags to set upon experiment creation

        :return: experiment_id (string) for the newly created experiment if successful, else None.
        """

        experiment_id = str(uuid.uuid4().hex)
        # Check if experiment already exists
        existing_experiment = self.get_experiment_by_name(name)
        if existing_experiment is not None:
            raise FileExistsError(f"Experiment by the name {name} already exists")

        experiment = Experiment(
            id=experiment_id,
            name=name,
            artifact_location=artifact_location,
            tags={tag.key: tag.value for tag in tags},
        )
        experiment.save()

        return experiment_id

    def get_experiment(self, experiment_id: str) -> mlflow.entities.Experiment:
        """
        Fetch the experiment by ID from the backend store.

        :param experiment_id: String id for the experiment

        :return: A single :py:class:`mlflow.entities.Experiment` object if it exists,
            otherwise raises an exception.

        """
        return Experiment.get(hash_key=experiment_id).to_mlflow()

    def get_experiment_by_name(
        self, experiment_name: str
    ) -> Optional[mlflow.entities.Experiment]:
        """
        Fetch the experiment by registered_model_name from the backend store.
        This is a base implementation using ``list_experiments``, derived classes may have
        some specialized implementations.

        :param experiment_name: Name of experiment

        :return: A single :py:class:`mlflow.entities.Experiment` object if it exists.
        """
        experiments = Experiment.name_index.query(hash_key=experiment_name)
        for experiment in experiments:
            return experiment.to_mlflow()

    def delete_experiment(self, experiment_id: str) -> str:
        """
        Delete the experiment from the backend store. Deleted experiments can be restored until
        permanently deleted.

        :param experiment_id: String id for the experiment
        """
        experiment = Experiment.get(experiment_id)
        experiment.lifecycle_stage = "deleted"
        experiment.save()
        return experiment.id

    def restore_experiment(self, experiment_id: str) -> str:
        """
        Restore deleted experiment unless it is permanently deleted.

        :param experiment_id: String id for the experiment
        """
        experiment = Experiment.get(experiment_id)
        experiment.lifecycle_stage = "active"
        experiment.save()
        return experiment.id

    def rename_experiment(self, experiment_id: str, new_name: str) -> str:
        """
        Update an experiment's registered_model_name. The new registered_model_name must be unique.

        :param experiment_id: String id for the experiment
        """
        experiment = Experiment.get(experiment_id)
        experiment.NAME = new_name
        experiment.save()
        return experiment.id

    def get_run(self, run_id: str) -> mlflow.entities.Run:
        """
        Fetch the run from backend store. The resulting :py:class:`Run <mlflow.entities.Run>`
        contains a collection of run metadata - :py:class:`RunInfo <mlflow.entities.RunInfo>`,
        as well as a collection of run parameters, tags, and metrics -
        :py:class`RunData <mlflow.entities.RunData>`. In the case where multiple metrics with the
        same key are logged for the run, the :py:class:`RunData <mlflow.entities.RunData>` contains
        the value at the latest timestamp for each metric. If there are multiple values with the
        latest timestamp for a given metric, the maximum of these values is returned.

        :param run_id: Unique identifier for the run.

        :return: A single :py:class:`mlflow.entities.Run` object, if the run exists. Otherwise,
                 raises an exception.
        """
        return Run.get(run_id).to_mlflow()

    def update_run_info(
        self, run_id: str, run_status: str, end_time: str
    ) -> mlflow.entities.RunInfo:
        """
        Update the metadata of the specified run.

        :return: :py:class:`mlflow.entities.RunInfo` describing the updated run.
        """
        run = Run.get(run_id)
        run.status = run_status
        run.end_time = end_time
        run.save()
        return run.to_mlflow().info

    def create_run(
        self,
        experiment_id: str,
        user_id: str,
        start_time: str,
        tags: List[mlflow.entities.RunTag],
    ) -> mlflow.entities.Run:
        """
        Create a run under the specified experiment ID, setting the run's status to "RUNNING"
        and the start time to the current time.

        :param experiment_id: String id of the experiment for this run
        :param user_id: ID of the user launching this run

        :return: The created Run object
        """
        run_id = str(uuid.uuid4().hex)

        experiment = Experiment.get(experiment_id)
        artifact_location = experiment.artifact_location
        if artifact_location is None:
            artifact_location = self.artifact_location

        run = Run(
            id=run_id,
            experiment_id=experiment_id,
            user_id=user_id,
            start_time=start_time,
            tags={tag.key: tag.value for tag in tags},
            artifact_uri=os.path.join(artifact_location, run_id, "artifacts"),
        )
        run.save()
        return run.to_mlflow()

    def delete_run(self, run_id: str) -> None:
        """
        Delete a run.

        :param run_id
        """
        run = Run.get(run_id)
        run.lifecycle_stage = "deleted"
        run.save()

    def restore_run(self, run_id: str) -> None:
        """
        Restore a run.

        :param run_id
        """
        run = Run.get(run_id)
        run.lifecycle_stage = "active"
        run.save()

    def set_experiment_tag(
        self, experiment_id: str, tag: mlflow.entities.ExperimentTag
    ) -> None:
        """
        Set a tag for the specified experiment

        :param experiment_id: String id for the experiment
        :param tag: :py:class:`mlflow.entities.ExperimentTag` instance to set
        """
        experiment = Experiment.get(experiment_id)
        tags = experiment.tags.as_dict()
        tags.update({tag.key: tag.value})
        experiment.tags = tags
        experiment.save()

    def get_metric_history(
        self, run_id: str, metric_key: str
    ) -> List[mlflow.entities.Metric]:
        """
        Return a list of metric objects corresponding to all values logged for a given metric.

        :param run_id: Unique identifier for run
        :param metric_key: Metric registered_model_name within the run

        :return: A list of :py:class:`mlflow.entities.Metric` entities if logged, else empty list
        """
        run = Run.get(run_id)
        metrics = []
        for metric in run.metrics:
            if metric.key == metric_key:
                metrics.append(metric.to_mlflow())
        return metrics

    def search_runs(
        self,
        experiment_ids: List[str],
        filter_string: str,
        run_view_type: Literal["ACTIVE_ONLY", "DELETED_ONLY", "ALL"],
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: List[str] = None,
        page_token: str = None,
    ) -> PagedList[mlflow.entities.Run]:
        """
        Return runs that match the given list of search expressions within the experiments.

        :param experiment_ids: List of experiment ids to scope the search
        :param filter_string: A search filter string.
        :param run_view_type: ACTIVE_ONLY, DELETED_ONLY, or ALL runs
        :param max_results: Maximum number of runs desired.
        :param order_by: List of order_by clauses.
        :param page_token: Token specifying the next page of results. It should be obtained from
            a ``search_runs`` call.

        :return: A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
            :py:class:`Run <mlflow.entities.Run>` objects that satisfy the search expressions.
            If the underlying tracking store supports pagination, the token for the next page may
            be obtained via the ``token`` attribute of the returned object; however, some store
            implementations may not support pagination and thus the returned token would not be
            meaningful in such cases.
        """
        runs, token = self._search_runs(
            experiment_ids,
            filter_string,
            run_view_type,
            max_results,
            order_by,
            page_token,
        )
        return PagedList(runs, token)

    def _search_runs(
        self,
        experiment_ids: List[str],
        filter_string: str,
        run_view_type: Literal["ACTIVE_ONLY", "DELETED_ONLY", "ALL"],
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: List[str] = None,
        page_token: str = None,
    ) -> Tuple[List[mlflow.entities.Run], Optional[str]]:
        """
        Return runs that match the given list of search expressions within the experiments, as
        well as a pagination token (indicating where the next page should start). Subclasses of
        ``AbstractStore`` should implement this method to support pagination instead of
        ``search_runs``.

        See ``search_runs`` for parameter descriptions.

        :return: A tuple of ``runs`` and ``token`` where ``runs`` is a list of
            :py:class:`mlflow.entities.Run` objects that satisfy the search expressions,
            and ``token`` is the pagination token for the next page of results.
        """

        if run_view_type == mlflow.entities.ViewType.ACTIVE_ONLY:
            filter_expression = Run.lifecycle_stage == "active"
        elif run_view_type == mlflow.entities.ViewType.DELETED_ONLY:
            filter_expression = Run.lifecycle_stage == "deleted"
        else:
            filter_expression = True
        if filter_string:
            filter_expression = filter_expression & Run.name.startswith(filter_string)
        runs = []
        for experiment_id in experiment_ids:
            runs.append(
                Run.scan((Run.experiment_id == experiment_id) & filter_expression)
            )

        return [run.to_mlflow() for run_list in runs for run in run_list], None

    def log_batch(
        self,
        run_id: str,
        metrics: List[mlflow.entities.Metric],
        params: List[mlflow.entities.Param],
        tags: List[mlflow.entities.RunTag],
    ) -> None:
        """
        Log multiple metrics, params, and tags for the specified run

        :param run_id: String id for the run
        :param metrics: List of :py:class:`mlflow.entities.Metric` instances to log
        :param params: List of :py:class:`mlflow.entities.Param` instances to log
        :param tags: List of :py:class:`mlflow.entities.RunTag` instances to log

        :return: None.
        """
        run = Run.get(run_id)
        # Update tags
        item_tags = run.tags.as_dict()
        item_tags.update({tag.key: tag.value for tag in tags or []})
        run.tags = item_tags
        # Update params
        item_params = run.params.as_dict()
        item_params.update({param.key: param.value for param in params or []})
        run.params = item_params
        # Update metrics
        item_metrics = run.metrics
        for metric in metrics or []:
            item_metrics.append(
                Metric(
                    key=metric.key,
                    value=metric.value,
                    timestamp=metric.timestamp,
                    step=metric.step,
                )
            )
        run.metrics = item_metrics
        run.save()

    def record_logged_model(self, run_id: str, mlflow_model: Any):
        """
        Record logged Model information with tracking store. The list of logged Model infos is
        maintained in a mlflow.models tag in JSON format.

        Note: The actual models are logged as artifacts via artifact repository.

        :param run_id: String id for the run
        :param mlflow_model: Model object to be recorded.

        The default implementation is a no-op.

        :return: None.
        """
        pass
