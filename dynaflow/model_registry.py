import json
import logging
import re
import time
from typing import List, Union

import mlflow
import mlflow.entities.model_registry
from mlflow.entities.model_registry.model_version_stages import (
    DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS,
    STAGE_ARCHIVED,
    STAGE_NONE,
    get_canonical_stage,
)
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.exceptions import MlflowException
from mlflow.store.entities import PagedList
from mlflow.store.model_registry.abstract_store import AbstractStore
from pynamodb.attributes import (
    DiscriminatorAttribute,
    MapAttribute,
    NumberAttribute,
    UnicodeAttribute,
)
from pynamodb.indexes import AllProjection, GlobalSecondaryIndex
from pynamodb.models import Model

log = logging.getLogger(__name__)


class VersionIndex(GlobalSecondaryIndex):
    class Meta:
        index_name = "mlflow-model-registry-version-index"
        region = "eu-central-1"
        projection = AllProjection()
        read_capacity_units = 2
        write_capacity_units = 1

    version: float = NumberAttribute(hash_key=True)
    registered_model_name: str = UnicodeAttribute(range_key=True)


class BaseEntry(Model):
    class Meta:
        table_name = "mlflow-model-registry"
        region = "eu-central-1"

    version_index = VersionIndex()

    cls: str = DiscriminatorAttribute()
    registered_model_name: str = UnicodeAttribute(hash_key=True)
    version: float = NumberAttribute(range_key=True)

    creation_timestamp: float = NumberAttribute()

    tags: MapAttribute = MapAttribute(default={})
    last_updated_timestamp: float = NumberAttribute(null=True)
    description: str = UnicodeAttribute(null=True)

    def evaluated_key(self) -> str:
        return json.dumps(
            dict(
                registered_model_name={"S": self.registered_model_name},
                version={"N": str(self.version)},
            )
        )


class ModelVersion(BaseEntry, discriminator="ModelVersion"):
    """
    DynamoDB abstraction for mlflow ModelVersions
    """

    current_stage: str = UnicodeAttribute(default_for_new=STAGE_NONE)
    status: str = UnicodeAttribute(
        default_for_new=ModelVersionStatus.to_string(ModelVersionStatus.READY)
    )

    user_id: str = UnicodeAttribute(null=True)
    source: str = UnicodeAttribute(null=True)
    run_id: str = UnicodeAttribute(null=True)
    status_message: str = UnicodeAttribute(null=True)
    run_link: str = UnicodeAttribute(null=True)

    def to_mlflow(self) -> mlflow.entities.model_registry.ModelVersion:
        return mlflow.entities.model_registry.ModelVersion(
            name=self.registered_model_name,
            version=self.version,
            user_id=self.user_id,
            current_stage=self.current_stage,
            source=self.source,
            run_id=self.run_id,
            status=self.status,
            status_message=self.status_message,
            run_link=self.run_link,
            creation_timestamp=int(self.creation_timestamp * 1000),
            last_updated_timestamp=int(self.last_updated_timestamp * 1000),
            tags=[
                mlflow.entities.model_registry.ModelVersionTag(key=key, value=value)
                for key, value in self.tags.as_dict().items()
            ],
            description=self.description,
        )


class RegisteredModel(BaseEntry, discriminator="RegisteredModel"):
    """
    DynamoDB abstraction for mlflow RegisteredModels
    """

    def load_versions(self) -> List[mlflow.entities.model_registry.ModelVersion]:
        versions = ModelVersion.query(hash_key=self.registered_model_name)
        return [version.to_mlflow() for version in versions]

    def to_mlflow(self) -> mlflow.entities.model_registry.RegisteredModel:
        return mlflow.entities.model_registry.RegisteredModel(
            name=self.registered_model_name,
            creation_timestamp=int(self.creation_timestamp * 1000),
            last_updated_timestamp=int(self.last_updated_timestamp * 1000),
            description=self.description,
            latest_versions=self.load_versions(),
            tags=[
                mlflow.entities.model_registry.RegisteredModelTag(key=key, value=value)
                for key, value in self.tags.as_dict().items()
            ],
        )


class DynamodbModelStore(AbstractStore):
    """
    Note:: Experimental: This entity may change or be removed in a future release without warning.
    Abstract class that defines API interfaces for storing Model Registry metadata.
    """

    def __init__(self, store_uri: str):
        super(DynamodbModelStore, self).__init__()

        _, region, tracking_table_name, model_table_name = store_uri.split(":")

        BaseEntry.Meta.table_name = model_table_name
        BaseEntry.Meta.region = region

        if not BaseEntry.exists():
            log.error(
                "Model Registry table does not exist."
                " Run 'dynaflow deploy' to deploy the necessary tables."
            )

    @staticmethod
    def _resolve_filter_condition(
        filter_string: str, filtered_model: Union[RegisteredModel, ModelVersion]
    ):
        if not re.match("name ilike '%.*%'", filter_string):
            raise ValueError(
                "Only filter string satisfying the regex "
                "pattern <name ilike '%.*%'> are allowed. "
                f"Received filter string: {filter_string}"
            )

        starts_with_str = re.sub("%'$", "", re.sub("^name ilike '%", "", filter_string))

        if starts_with_str:
            range_key_condition = filtered_model.registered_model_name.startswith(
                starts_with_str
            )
        else:
            range_key_condition = None

        return range_key_condition

    @staticmethod
    def _query_result_to_paged_list(
        query_result,
    ) -> PagedList[
        Union[
            mlflow.entities.model_registry.RegisteredModel,
            mlflow.entities.model_registry.ModelVersion,
        ]
    ]:
        registered_models = [model for model in query_result]

        if not registered_models:
            return PagedList([], None)

        return PagedList(
            [model.to_mlflow() for model in registered_models],
            registered_models[-1].evaluated_key(),
        )

    # CRUD API for RegisteredModel objects

    def create_registered_model(
        self,
        name: str,
        tags: List[mlflow.entities.model_registry.RegisteredModelTag] = None,
        description: str = None,
    ) -> mlflow.entities.model_registry.RegisteredModel:
        """
        Create a new registered model in backend store.

        :param name: Name of the new model. This is expected to be unique in the backend store.
        :param tags: A list of :py:class:`mlflow.entities.model_registry.RegisteredModelTag`
                     instances associated with this registered model.
        :param description: Description of the model.
        :return: A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
                 created in the backend.
        """
        creation_timestamp = time.time()
        registered_model = RegisteredModel(
            registered_model_name=name,
            version=-1,
            tags={tag.key: tag.value for tag in tags},
            description=description,
            last_updated_timestamp=creation_timestamp,
            creation_timestamp=creation_timestamp,
        )
        registered_model.save()
        return registered_model.to_mlflow()

    def update_registered_model(self, name: str, description: str):
        """
        Update description of the registered model.

        :param name: Registered model name.
        :param description: New description.
        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        registered_model = RegisteredModel.get(hash_key=name, range_key=-1)
        registered_model.description = description
        registered_model.last_updated_timestamp = time.time()
        registered_model.save()
        return registered_model.to_mlflow()

    def rename_registered_model(
        self, name: str, new_name: str
    ) -> mlflow.entities.model_registry.RegisteredModel:
        """
        Rename the registered model.

        :param name: Registered model name.
        :param new_name: New proposed name.
        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        registered_model = RegisteredModel.get(hash_key=name, range_key=-1)
        registered_model.registered_model_name = new_name
        registered_model.last_updated_timestamp = time.time()

        # Update all associated versions to reflect the new model Name
        registered_versions = ModelVersion.query(hash_key=name)
        for version in registered_versions:
            version.registered_model_name = name
            version.last_updated_timestamp = time.time()
            version.save()

        registered_model.save()

        return registered_model.to_mlflow()

    def delete_registered_model(self, name: str) -> None:
        """
        Delete the registered model.
        Backend raises exception if a registered model with given name does not exist.

        :param name: Registered model name.
        :return: None
        """
        registered_model = RegisteredModel.get(hash_key=name, range_key=-1)
        registered_model.delete()
        registered_versions = RegisteredModel.query(hash_key=name)
        for version in registered_versions:
            version.delete()

    def list_registered_models(
        self, max_results: int, page_token: str
    ) -> PagedList[mlflow.entities.model_registry.RegisteredModel]:
        """
        List of all registered models.

        :param max_results: Maximum number of registered models desired.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``list_registered_models`` call.
        :return: A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
                that satisfy the search expressions. The pagination token for the next page can be
                obtained via the ``token`` attribute of the object.
        """
        if page_token:
            page_token = json.loads(page_token)

        registered_models = RegisteredModel.version_index.query(
            hash_key=-1, last_evaluated_key=page_token
        )
        return self._query_result_to_paged_list(registered_models)

    def search_registered_models(
        self,
        filter_string: str = None,
        max_results: int = None,
        order_by: str = None,
        page_token: str = None,
    ) -> PagedList[mlflow.entities.model_registry.RegisteredModel]:
        """
        Search for registered models in backend that satisfy the filter criteria.

        :param filter_string: Filter query string, defaults to searching all registered models.
        :param max_results: Maximum number of registered models desired.
        :param order_by: List of column names with ASC|DESC annotation, to be used for ordering
                         matching search results.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``search_registered_models`` call.
        :return: A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
                that satisfy the search expressions. The pagination token for the next page can be
                obtained via the ``token`` attribute of the object.
        """
        # We can only filter using the startswith expression so we use it as a drop-in for the
        # ilike filter clause
        range_key_condition = self._resolve_filter_condition(
            filter_string, RegisteredModel
        )

        if page_token:
            page_token = json.loads(page_token)

        registered_models = RegisteredModel.version_index.query(
            hash_key=-1,
            range_key_condition=range_key_condition,
            last_evaluated_key=page_token,
            limit=max_results,
        )
        return self._query_result_to_paged_list(registered_models)

    def get_registered_model(
        self, name: str
    ) -> mlflow.entities.model_registry.RegisteredModel:
        """
        Get registered model instance by name.

        :param name: Registered model name.
        :return: A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        registered_model = RegisteredModel.get(hash_key=name, range_key=-1)
        return registered_model.to_mlflow()

    def get_latest_versions(
        self, name: str, stages: List[str] = None
    ) -> List[mlflow.entities.model_registry.ModelVersion]:
        """
        Latest version models for each requested stage. If no ``stages`` argument is provided,
        returns the latest version for each stage.

        :param name: Registered model name.
        :param stages: List of desired stages. If input list is None, return latest versions for
                       for 'Staging' and 'Production' stages.
        :return: List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        all_versions = ModelVersion.query(hash_key=name)
        latest_by_stage = {}
        for version in all_versions:
            try:
                if version.version > latest_by_stage[version.current_stage].version:
                    latest_by_stage[version.current_stage] = version
            except KeyError:
                latest_by_stage[version.current_stage] = version

        if stages:
            latest_by_stage = {
                key: value for key, value in latest_by_stage.items() if key in stages
            }

        return list(latest_by_stage.values())

    def set_registered_model_tag(
        self, name: str, tag: mlflow.entities.model_registry.RegisteredModelTag
    ) -> None:
        """
        Set a tag for the registered model.

        :param name: Registered model name.
        :param tag: :py:class:`mlflow.entities.model_registry.RegisteredModelTag` instance to log.
        :return: None
        """
        registered_model = RegisteredModel.get(hash_key=name, range_key=-1)
        current_tags = registered_model.tags.as_dict()
        current_tags.update({tag.key: tag.value})
        registered_model.tags = current_tags
        registered_model.save()

    def delete_registered_model_tag(self, name: str, key: str) -> None:
        """
        Delete a tag associated with the registered model.

        :param name: Registered model name.
        :param key: Registered model tag key.
        :return: None
        """
        registered_model = RegisteredModel.get(hash_key=name, range_key=-1)
        registered_model.tags.pop(name)
        registered_model.save()

    # CRUD API for ModelVersion objects

    def create_model_version(
        self,
        name: str,
        source: str,
        run_id=None,
        tags: List[mlflow.entities.model_registry.ModelVersionTag] = None,
        run_link: str = None,
        description: str = None,
    ) -> mlflow.entities.model_registry.ModelVersion:
        """
        Create a new model version from given source and run ID.

        :param name: Registered model name.
        :param source: Source path where the MLflow model is stored.
        :param run_id: Run ID from MLflow tracking server that generated the model.
        :param tags: A list of :py:class:`mlflow.entities.model_registry.ModelVersionTag`
                     instances associated with this model version.
        :param run_link: Link to the run from an MLflow tracking server that generated this model.
        :param description: Description of the version.
        :return: A single object of :py:class:`mlflow.entities.model_registry.ModelVersion`
                 created in the backend.
        """

        # Check how many versions are there and increment by 1
        new_version = ModelVersion.query(name).total_count + 1
        creation_timestamp = time.time()
        model_version = ModelVersion(
            registered_model_name=name,
            version=new_version,
            source=source,
            tags={tag.key: tag.value for tag in tags},
            run_link=run_link,
            run_id=run_id,
            description=description,
            creation_timestamp=creation_timestamp,
            last_updated_timestamp=creation_timestamp,
        )
        model_version.save()
        return model_version.to_mlflow()

    def update_model_version(
        self, name: str, version: int, description
    ) -> mlflow.entities.model_registry.ModelVersion:
        """
        Update metadata associated with a model version in backend.

        :param name: Registered model name.
        :param version: Registered model version.
        :param description: New model description.
        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        model_version = ModelVersion.get(hash_key=name, range_key=int(version))
        model_version.description = description
        model_version.last_updated_timestamp = time.time()
        model_version.save()
        return model_version.to_mlflow()

    def transition_model_version_stage(
        self, name: str, version: int, stage: str, archive_existing_versions: bool
    ) -> mlflow.entities.model_registry.ModelVersion:
        """
        Update model version stage.

        :param name: Registered model name.
        :param version: Registered model version.
        :param new_stage: New desired stage for this model version.
        :param archive_existing_versions: If this flag is set to ``True``, all existing model
            versions in the stage will be automically moved to the "archived" stage. Only valid
            when ``stage`` is ``"staging"`` or ``"production"`` otherwise an error will be raised.

        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """

        is_active_stage = (
            get_canonical_stage(stage) in DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS
        )
        if archive_existing_versions and not is_active_stage:
            msg_tpl = (
                "Model version transition cannot archive existing model versions "
                "because '{}' is not an Active stage. Valid stages are {}"
            )
            raise MlflowException(
                msg_tpl.format(stage, DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS)
            )

        version = int(version)

        model_version = ModelVersion.get(hash_key=name, range_key=version)
        model_version.current_stage = get_canonical_stage(stage)

        if archive_existing_versions:
            other_versions = ModelVersion.query(
                hash_key=name, range_key_condition=ModelVersion.version < version
            )
            with ModelVersion.batch_write() as batch:
                for version in other_versions:
                    version.current_stage = STAGE_ARCHIVED
                    batch.save(version)

        model_version.save()

        return model_version.to_mlflow()

    def delete_model_version(self, name: str, version: int) -> None:
        """
        Delete model version in backend.

        :param name: Registered model name.
        :param version: Registered model version.
        :return: None
        """
        model_version = ModelVersion.get(hash_key=name, range_key=int(version))
        model_version.delete()

    def get_model_version(
        self, name: str, version: int
    ) -> mlflow.entities.model_registry.ModelVersion:
        """
        Get the model version instance by name and version.

        :param name: Registered model name.
        :param version: Registered model version.
        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        model_version = ModelVersion.get(hash_key=name, range_key=int(version))
        return model_version.to_mlflow()

    def get_model_version_download_uri(self, name: str, version: int) -> str:
        """
        Get the download location in Model Registry for this model version.
        NOTE: For first version of Model Registry, since the models are not copied over to another
              location, download URI points to input source path.

        :param name: Registered model name.
        :param version: Registered model version.
        :return: A single URI location that allows reads for downloading.
        """
        model_version = ModelVersion.get(hash_key=name, range_key=int(version))
        return model_version.source

    def search_model_versions(
        self, filter_string: str
    ) -> PagedList[mlflow.entities.model_registry.ModelVersion]:
        """
        Search for model versions in backend that satisfy the filter criteria.

        :param filter_string: A filter string expression. Currently supports a single filter
                              condition either name of model like ``name = 'model_name'`` or
                              ``run_id = '...'``.
        :return: PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion`
                 objects.
        """

        if re.match("^.*='.*'$", filter_string):
            model_name = re.sub("'$", "", re.sub("^.*='", "", filter_string))
            model_versions = ModelVersion.query(hash_key=model_name)
        else:
            try:
                model_name_condition = self._resolve_filter_condition(
                    filter_string, ModelVersion
                )
                model_versions = ModelVersion.scan(model_name_condition)
            except ValueError:
                raise (f"Filter expression not supported. {filter_string}")

        return self._query_result_to_paged_list(model_versions)

    def set_model_version_tag(
        self,
        name: str,
        version: int,
        tag: mlflow.entities.model_registry.ModelVersionTag,
    ) -> None:
        """
        Set a tag for the model version.

        :param name: Registered model name.
        :param version: Registered model version.
        :param tag: :py:class:`mlflow.entities.model_registry.ModelVersionTag` instance to log.
        :return: None
        """
        model_version = ModelVersion.get(hash_key=name, range_key=int(version))
        tags = model_version.tags.as_dict()
        tags.update({tag.key: tag.value})
        model_version.tags = tags
        model_version.last_updated_timestamp = time.time()
        model_version.save()

    def delete_model_version_tag(self, name: str, version: int, key: str) -> None:
        """
        Delete a tag associated with the model version.

        :param name: Registered model name.
        :param version: Registered model version.
        :param key: Tag key.
        :return: None
        """
        model_version = ModelVersion.get(hash_key=name, range_key=int(version))
        model_version.tags.pop(key)
        model_version.save()
