# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Classes for working with language models."""

import dataclasses
import json
import os
from typing import Any, Dict, List, Optional, Union

from google.cloud import storage

from google.cloud import aiplatform
from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer as aiplatform_initializer
from google.cloud.aiplatform import utils as aiplatform_utils
from google.cloud.aiplatform.utils import gcs_utils

from google.cloud.aiplatform.compat.services import (
    model_garden_service_client,
)
from google.cloud.aiplatform.compat.types import (
    pipeline_state as gca_pipeline_state,
)

try:
    import pandas
except ImportError:
    pandas = None


_LOGGER = base.Logger(__name__)

# Model Evaluation constants
_TEXT_CLASSIFICATION_TASK_NAME = "text-classification"
_TEXT_GENERATION_TASK_NAME = "text-generation"
_QA_TASK_NAME = "question-answering"
_SUMMARIZATION_TASK_NAME = "summarization"

_EVALUATION_TASKS = frozenset(
    [
        _TEXT_CLASSIFICATION_TASK_NAME,
        _TEXT_GENERATION_TASK_NAME,
        _QA_TASK_NAME,
        _SUMMARIZATION_TASK_NAME,
    ]
)


_TEXT_CLASSIFICATION_TEMPLATE_URL = "https://us-kfp.pkg.dev/vertex-evaluation/pipeline-templates/evaluation-llm-classification-pipeline"
_TEXT_GENERATION_QA_SUMMARIZATION_TEMPLATE_URL = "https://us-kfp.pkg.dev/vertex-evaluation/pipeline-templates/evaluation-llm-text-generation-pipeline"

_EVALUATION_TEMPLATE_VERSION_TAG = "1.0.1"

_EVALUATION_TEMPLATE_URLS = {
    _TEXT_CLASSIFICATION_TASK_NAME: f"{_TEXT_CLASSIFICATION_TEMPLATE_URL}/{_EVALUATION_TEMPLATE_VERSION_TAG}",
    _TEXT_GENERATION_TASK_NAME: f"{_TEXT_GENERATION_QA_SUMMARIZATION_TEMPLATE_URL}/{_EVALUATION_TEMPLATE_VERSION_TAG}",
    _QA_TASK_NAME: f"{_TEXT_GENERATION_QA_SUMMARIZATION_TEMPLATE_URL}/{_EVALUATION_TEMPLATE_VERSION_TAG}",
    _SUMMARIZATION_TASK_NAME: f"{_TEXT_GENERATION_QA_SUMMARIZATION_TEMPLATE_URL}/{_EVALUATION_TEMPLATE_VERSION_TAG}",
}


_EVALUATION_PIPELINE_COMPONENT_IDENTIFIER = "fpc-llm-evaluation"

# TODO: update this when BP removes the input size limit
_BATCH_PREDICTION_ROW_LIMIT = 1000

_EVAL_SUPPORTED_BASE_MODELS = ["text-bison@001"]


def _check_dataset_is_within_size_limit(
    data: "pandas.DataFrame",
) -> None:

    if len(data) < _BATCH_PREDICTION_ROW_LIMIT:
        return

    raise ValueError(
        f"Your evaluation dataset size exceeds the limit of {_BATCH_PREDICTION_ROW_LIMIT}"
    )


def _get_model_resource_name_and_validate(
    model_name: str,
) -> str:
    """Returns the resource name string for the model.

    Model Registry resource names will stay the same. For Publisher Models, we need to
    pass the full resource name to the evaluation template and ensure the base model
    supports evaluation.

    Args:
        model_name (str):
            Required. The full resource name of the Model Registry model or base publisher model to run evaluation on.

    Returns:
        The formatted model_name string.

    Raises:
        ValueError
            If a base PublisherModel was provided and the model doesn't support evaluation.
    """

    if "publishers/" not in model_name:
        # Model Registry resource
        return model_name

    else:
        # rpartition is used here since `model_name` is a fully qualified PublisherModel resource name (with project ID) and parse_publisher_model_path takes only the `publishers/...` path as input
        publisher_model_parts = model_garden_service_client.ModelGardenServiceClient.parse_publisher_model_path(
            "".join(model_name.rpartition("publishers")[1:])
        )

        if publisher_model_parts and publisher_model_parts["publisher"] == "google":
            model_id = publisher_model_parts["model"]

            if model_id not in _EVAL_SUPPORTED_BASE_MODELS:
                raise ValueError(
                    f"The provided model {model_name} does not support evaluation."
                )

            # The full value of model_name (`projects/{project}/locations/{location}/publishers/google/models/{model_id}`) is not supported by BP
            return f"publishers/google/models/{model_id}"

        raise ValueError(
            f"The provided model {model_name} does not support evaluation."
        )


def _get_template_url(task_name: str) -> Optional[str]:
    """Returns the pipeline template to use for the evaluation task.

    Args:
        task_name (str):
            Required. The name of the evaluation task to run.

    Returns:
        The evaluation pipeline template path.
    """

    return _EVALUATION_TEMPLATE_URLS.get(task_name)


@dataclasses.dataclass
class _EvaluationTaskSpec:
    """Base class for task-specific model evaluation configuration parameters.

    This class should not be instantiated directly, instead use the subclass corresponding
    to your evaluation task.

    Args:
        ground_truth_data (Union[List[str], str, pandas.DataFrame]):
            Required. The ground truth data to use for this evaluation job. This can be
            either a Pandas DataFrame, a Cloud Storage URI of your JSONL data file, or a list of multiple
            JSONL files on Cloud Storage.

    Raises:
        ValueError:
            If task_spec.ground_truth_data is formatted incorrectly.
            If task_spec.ground_truth_data is a Pandas DataFrame and exceeds 1000 rows.
            If task_spec.ground_truth_data is not a string, list, or Pandas DataFrame.
    """

    ground_truth_data: Union[List[str], str, "pandas.DataFrame"]

    @property
    def task_name(self) -> str:
        pass

    def __post_init__(self):

        if isinstance(self.ground_truth_data, str):
            self.ground_truth_data = [self.ground_truth_data]

        if isinstance(self.ground_truth_data, list) and not all(
            item.startswith("gs://") for item in self.ground_truth_data
        ):
            raise ValueError("Please provide a valid GCS URI starting with 'gs://'")

        if pandas and isinstance(self.ground_truth_data, pandas.DataFrame):

            _check_dataset_is_within_size_limit(self.ground_truth_data)


@dataclasses.dataclass
class EvaluationTextClassificationSpec(_EvaluationTaskSpec):
    """Spec for text classification model evaluation tasks.

    Args:
        target_column_name (str):
            Required. The label column in the dataset provided in `ground_truth_data`. Required when task_name='text-classification'.
        class_names (List[str]):
            Required. A list of all possible label names in your dataset. Required when task_name='text-classification'.
    """

    target_column_name: str
    class_names: List[str]

    @property
    def task_name(self) -> str:
        return "text-classification"


@dataclasses.dataclass
class EvaluationTextGenerationSpec(_EvaluationTaskSpec):
    """Spec for text generation model evaluation tasks."""

    @property
    def task_name(self) -> str:
        return "text-generation"


@dataclasses.dataclass
class EvaluationQuestionAnsweringSpec(_EvaluationTaskSpec):
    """Spec for question answering model evaluation tasks."""

    task_name: str = "question-answering"


@dataclasses.dataclass
class EvaluationTextSummarizationSpec(_EvaluationTaskSpec):
    """Spec for text summarization model evaluation tasks."""

    task_name: str = "summarization"


@dataclasses.dataclass
class EvaluationMetricResponse:
    """The evaluation metric response.

    Args:
        bleu (float):
            Optional. BLEU (Bilingual evauation understudy). Scores based on sacrebleu implementation.
        rougeLSum (float):
            Optional. ROUGE-L (Longest Common Subsequence) scoring at summary level.
    """

    bleu: Optional[float] = None
    rougeLSum: Optional[float] = None


@dataclasses.dataclass
class EvaluationClassificationMetricResponse:
    """The evaluation metric response for classification metrics.

    Args:
        label_name (str):
            Optional. The name of the label associated with the metrics. This is only
            returned when `only_summary_metrics=False` is passed to evaluate().
        auPrc (float):
            Optional. The area under the precision recall curve.
        auRoc (float):
            Optional. The area under the receiver operating characteristic curve.
        logLoss (float):
            Optional. Logarithmic loss.
        confidenceMetrics (List[Dict[str, Any]]):
            Optional. This is only returned when `only_summary_metrics=False` is
            passed to evaluate().
        confusionMatrix (Dict[str, Any]):
          Optional. This is only returned when `only_summary_metrics=False` is
          passed to evaluate().
    """

    label_name: Optional[str] = None
    auPrc: Optional[float] = None
    auRoc: Optional[float] = None
    logLoss: Optional[float] = None
    confidenceMetrics: Optional[List[Dict[str, Any]]] = None
    confusionMatrix: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class ListEvaluationMetricsResponse:
    """The response for list_evaluation_metrics().

    Args:
        metrics (Union[EvaluationMetricResponse, EvaluationClassificationMetricResponse, List[EvaluationClassificationMetricResponse]]):
            The metrics for the evaluation. If `only_summary_metrics=False` is passed to
            `list_evaluation_metrics()` and the evaluation task is 'text-classification', this will be of type
            List[EvaluationClassificationMetricResponse] for the given evaluation.
        input_dataset_paths (str):
            The Google Cloud Storage paths to the dataset used for this evaluation.
        evaluation_task (str):
            The type of evaluation task for the evaluation.
    """

    metrics: Union[
        EvaluationMetricResponse,
        EvaluationClassificationMetricResponse,
        List[EvaluationClassificationMetricResponse],
    ]
    input_dataset_paths: str
    evaluation_task: str


def _populate_eval_template_params(
    task_spec: _EvaluationTaskSpec,
    model_name: str,
    service_account: Optional[str] = None,
    machine_type: Optional[str] = None,
    network: Optional[str] = None,
    encryption_spec_key_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Populates a dictionary of template parameters for the evaluation PipelineJob.

    Args:
        task_spec (EvaluationTaskSpec):
            The EvaluationTaskSpec passed to evaluate() for this job
        model_name (str):
            The resource name of the model being evaluated. Either a PublisherModel or
            ModelRegistry resource name.
        service_account (Optional[str]):
            The default service account for workload run-as account.
        machine_type (Optional[str]):
            Optional. The type of the machine to run the evaluation job on.
        network (Optional[str]):
            Optional.
        encryption_spec_key_name (Optional[str]):
            Optional.

    Returns:
        Dict[str, Any]:
            A dictionary of template parameter names and values to be passed to the PipelineJob
            running the model evaluation.
    """

    ground_truth_data_gcs_path = task_spec.ground_truth_data

    if isinstance(task_spec.ground_truth_data, pandas.DataFrame):

        staging_bucket = aiplatform_initializer.global_config.staging_bucket

        if not staging_bucket:
            staging_bucket = (
                gcs_utils.create_gcs_bucket_for_pipeline_artifacts_if_it_does_not_exist()
            )

        # Convert to jsonl file and upload to gcs
        dataset_uri = os.path.join(
            staging_bucket,
            f"evaluation_data_{aiplatform_utils.timestamped_unique_name()}",
            "eval_data.jsonl",
        )

        gcs_utils._upload_pandas_df_to_gcs(
            df=task_spec.ground_truth_data, upload_gcs_path=dataset_uri
        )
        ground_truth_data_gcs_path = [dataset_uri]

    template_params = {
        "project": aiplatform_initializer.global_config.project,
        "location": aiplatform_initializer.global_config.location,
        "batch_predict_gcs_destination_output_uri": aiplatform_initializer.global_config.staging_bucket
        or gcs_utils.create_gcs_bucket_for_pipeline_artifacts_if_it_does_not_exist(),
        "model_name": model_name,
        "batch_predict_gcs_source_uris": ground_truth_data_gcs_path,
        "service_account": service_account,
        "machine_type": machine_type,
        "encrytion_spec_key_name": encryption_spec_key_name
        or aiplatform_initializer.global_config.encryption_spec_key_name,
        "network": network or aiplatform_initializer.global_config.network,
    }

    if task_spec.task_name == _TEXT_CLASSIFICATION_TASK_NAME:
        template_params["evaluation_class_labels"] = task_spec.class_names
        template_params["target_field_name"] = task_spec.target_column_name
    else:
        template_params["evaluation_task"] = task_spec.task_name

    return template_params


# TODO (b/285947054): update to use public pipeline contract
def _get_gcs_uri_from_pipeline_task_details(
    pipeline_job: aiplatform.PipelineJob,
) -> Optional[str]:
    """Gets the GCS URI from the PipelineJob output.

    Args:
        pipeline_job (aiplatform.PipelineJob)
            The PipelineJob resource to get the metrics GCS URI from

    Returns:
        The GCS URI of the evaluation metrics as a string.
    """

    for task in pipeline_job.task_details:
        if task.task_name == pipeline_job.name and "evaluation_metrics" in task.outputs:
            return task.outputs["evaluation_metrics"].artifacts[0].uri


def _convert_metrics_dict_to_response_type(
    metrics: Dict[str, Any],
    metric_name: str,
) -> EvaluationClassificationMetricResponse:
    metrics_response = EvaluationClassificationMetricResponse(label_name=metric_name)
    metric_names = list(
        EvaluationClassificationMetricResponse.__dataclass_fields__.keys()
    )
    for metric, value in metrics.items():
        if metric in metric_names:
            setattr(metrics_response, metric, value)
    return metrics_response


def _format_classification_metrics(
    metrics: Dict[str, Any]
) -> List[EvaluationClassificationMetricResponse]:
    """Reformats classification metrics returned by the eval pipeline to make them more readable.

    Returned metrics dictionary includes one key named 'overall' with the metrics for all data, along
    with a key for each label in the dataset with specific metrics for data with that label.

    Example schema of reformatted metrics:

    {
        EvaluationClassificationMetricResponse(
            label_name="overall",
            auPrc=...,
            ...
        ),
        EvaluationClassificationMetricResponse(
            label_name="label_1",
            auPrc=...,
            ...
        ),
        EvaluationClassificationMetricResponse(
            label_name="label_2",
            auPrc=...,
            ...
        )
    """

    reformatted_metrics = []

    # TODO: see if we can do this without relying on specific keys, i.e. slicedMetrics

    # First add overall metrics
    reformatted_metrics.append(
        _convert_metrics_dict_to_response_type(
            metrics=metrics["slicedMetrics"][0]["metrics"]["classification"],
            metric_name="overall",
        )
    )

    # Then add metrics for each slice
    for idx in range(1, len(metrics["slicedMetrics"])):
        metric_slice_name = metrics["slicedMetrics"][idx]["singleOutputSlicingSpec"][
            "value"
        ]
        reformatted_metrics.append(
            _convert_metrics_dict_to_response_type(
                metrics=metrics["slicedMetrics"][idx]["metrics"]["classification"],
                metric_name=metric_slice_name,
            )
        )

    return reformatted_metrics


def _get_metrics_from_gcs_uri(gcs_uri: str) -> Dict[str, Any]:
    """Downloads evaluation metrics from GCS path."""

    storage_client = storage.Client(
        credentials=aiplatform_initializer.global_config.credentials
    )

    gcs_uri_split = gcs_uri[5:].split("/")
    bucket_name = gcs_uri_split[0]

    blob_name = "/".join(gcs_uri_split[1:])
    bucket = storage_client.bucket(bucket_name)
    blob_bytes = bucket.blob(blob_name).download_as_bytes().decode("utf-8")
    metrics_json = json.loads(blob_bytes)

    # Sliced classification metrics case, format data
    if "slicedMetrics" in metrics_json:
        return _format_classification_metrics(metrics_json)
    # If classification metrics don't contain slices, use EvaluationClassificationMetricResponse type
    elif "auPrc" in metrics_json:
        metrics_response = EvaluationClassificationMetricResponse()
        metric_names = list(
            EvaluationClassificationMetricResponse.__dataclass_fields__.keys()
        )
    # All other metric types
    else:
        metrics_response = EvaluationMetricResponse()
        metric_names = list(EvaluationMetricResponse.__dataclass_fields__.keys())

    for metric, value in metrics_json.items():
        if metric in metric_names:
            setattr(metrics_response, metric, value)
    return metrics_response


def _get_metrics_from_pipeline_task_details(
    pipeline_job: aiplatform.PipelineJob,
) -> Union[EvaluationMetricResponse, EvaluationClassificationMetricResponse]:
    """Gets the evaluation metrics from the PipelineJob TaskDetails.

    Args:
        pipeline_job (aiplatform.PipelineJob)
            The PipelineJob resource to get the metrics from

    Returns:
        A dictionary with the evaluation metrics
    """
    metrics = {}

    # TODO (b/292076101): this now uses a public pipelines contract, but still relies on task_details
    for task in pipeline_job.task_details:
        if task.task_name == pipeline_job.name:
            for output in task.outputs:
                for metric_name, metric_value in (
                    task.outputs[output].artifacts[0].metadata.items()
                ):
                    metrics[metric_name] = metric_value

            if "auPrc" in metrics:
                metrics_response = EvaluationClassificationMetricResponse()
                metric_names = list(
                    EvaluationClassificationMetricResponse.__dataclass_fields__.keys()
                )
            else:
                metrics_response = EvaluationMetricResponse()
                metric_names = list(
                    EvaluationMetricResponse.__dataclass_fields__.keys()
                )

            for metric, value in metrics.items():
                if metric in metric_names:
                    setattr(metrics_response, metric, value)
            return metrics_response


class _LanguageModelEvaluationJob:
    """Represents a model evaluation job for LLM models.

    These evaluation jobs are run as a Vertex Pipeline.
    """

    def result(self, only_summary_metrics: bool) -> Optional[Dict[str, Any]]:
        """Blocks on completion of the model evaluation PipelineJob and returns metrics."""

        self._pipeline_job.wait()

        if only_summary_metrics:
            return _get_metrics_from_pipeline_task_details(self._pipeline_job)
        else:
            gcs_uri = _get_gcs_uri_from_pipeline_task_details(self._pipeline_job)
            if gcs_uri:
                return _get_metrics_from_gcs_uri(gcs_uri)

    def __init__(
        self,
        pipeline_job: aiplatform.PipelineJob,
    ):
        self._pipeline_job = pipeline_job


class _EvaluatableLanguageModel:
    """Mixin class for LLMs that support model evaluation."""

    # TODO (b/282975912): convert training job specific args to a TrainingConfig
    def evaluate(
        self,
        task_spec: _EvaluationTaskSpec,
        only_summary_metrics: Optional[bool] = True,
        service_account: Optional[str] = None,
        machine_type: Optional[str] = None,
        network: Optional[str] = None,
        encryption_spec_key_name: Optional[str] = None,
    ) -> Union[
        EvaluationMetricResponse,
        EvaluationClassificationMetricResponse,
        List[EvaluationClassificationMetricResponse],
    ]:
        """Runs model evaluation using the provided input and ground truth data.

        This creates an evaluation job and blocks until the job completes, about
        10 - 20 minutes.

        Example:
        ```
        model = TextGenerationModel.from_pretrained("text-bison@001")
        eval_metrics = model.evaluate(
            task_spec=EvaluationTextGenerationSpec(
                ground_truth_data="gs://my-bucket/ground-truth.jsonl",
            )
        )
        ```

        Args:
            task_spec (_EvaluationTaskSpec):
                Required. The configuration spec for your model evaluation job. Choose the spec corresponding
                with the evaluation task you are performing, one of: EvaluationClassificationSpec, EvaluationTextGenerationSpec,
                EvaluationTextSummarizationSpec, EvaluationQuestionAnsweringSpec.

                For example, a valid classification `task_spec` is:
                EvaluationTextClassificationSpec(
                    ground_truth_data=["gs://bucket/path/to/your/data.jsonl"],
                    class_names=["cheddar", "gouda", "camembert"],
                    target_column_name="cheese_type",
                )
            only_summary_metrics (bool):
                Optional. Setting this field to False only affects the metrics returned for text classification tasks.
                When False, text classification metrics will include additional sliced metrics fields, with metrics for
                each label slice in the data.
            service_account (str):
                Optional. Sets the default service account for workload run-as account. The service account
                running the pipeline submitting jobs must have act-as permission on this run-as account. If not
                provided, the default Compute Engine service account will be used.
            machine_type (str):
                Optional. The type of the machine to run the evaluation job on. The default value is "e2-highmem-16". For
                tasks with a large evaluation dataset, a bigger machine type may be required.
                For more details about this input config, see
                https://cloud.google.com/vertex-ai/docs/training/configure-compute#machine-types.
            network (str):
                The full name of the Compute Engine network to which the job
                should be peered. For example, projects/12345/global/networks/myVPC.
                Format is of the form projects/{project}/global/networks/{network}.
                Where {project} is a project number, as in 12345, and {network} is a
                network name. Private services access must already be configured for the
                network. If left unspecified, the job is not peered with any network.
            encryption_spec_key_name (str):
                Customer-managed encryption key options for the
                CustomJob. If this is set, then all resources created by the CustomJob
                will be encrypted with the provided encryption key.

        Returns:
            Union[EvaluationMetricResponse, EvaluationClassificationMetricResponse, List[EvaluationClassificationMetricResponse]]
                The evaluation metrics from this evaluation job. When `only_summary_metrics=False` is passed
                and the evaluation task type is 'text-classification', the return type will be List[EvaluationClassificationMetricResponse],
                where each value in the list is the metrics associated with a particular classification label.
        """

        model_name = _get_model_resource_name_and_validate(self._model_resource_name)

        template_params = _populate_eval_template_params(
            task_spec=task_spec,
            model_name=model_name,
            service_account=service_account,
            machine_type=machine_type,
            network=network,
            encryption_spec_key_name=encryption_spec_key_name,
        )

        template_path = _get_template_url(task_spec.task_name)

        pipeline_job = aiplatform.PipelineJob(
            template_path=template_path,
            parameter_values=template_params,
            display_name=f"llm-eval-sdk-{aiplatform_utils.timestamped_unique_name()}",
        )
        pipeline_job.submit()

        eval_job = _LanguageModelEvaluationJob(pipeline_job=pipeline_job)

        _LOGGER.info(
            "Your evaluation job is running and will take 15-20 minutes to complete. Click on the PipelineJob link to view progress."
        )

        # NOTE: only_summary_metrics is passed because getting metrics from the artifact is faster than downloading from GCS
        # GCS is only needed for additional metrics for text-classification tasks
        return eval_job.result(only_summary_metrics=only_summary_metrics)

    def list_evaluation_metrics(
        self,
        task_name: Optional[str] = None,
        only_summary_metrics: Optional[bool] = True,
    ) -> List[ListEvaluationMetricsResponse]:
        """Lists the evaluation metrics from all evaluation jobs run on this model.

        Args:
            task_name (str):
                Optional. The task name to return evaluation metrics for. If provided, this will only return evaluation
                metrics for tasks of the provided type. This matches the possible values passed to EvaluationTaskType.task_name,
                and must be one of 'text-generation', 'text-classification', 'summarization', or 'question-answering'.

        Returns:
            Dict[str, Any]
                The evaluation metrics from all evaluation jobs run on this model.

        """

        model_name = self._model_resource_name

        publisher_model_parts = model_garden_service_client.ModelGardenServiceClient.parse_publisher_model_path(
            "".join(model_name.rpartition("publishers")[1:])
        )

        if publisher_model_parts:
            model_id = publisher_model_parts["model"]
            model_name = f"publishers/google/models/{model_id}"

        filters = f'metadata.component_type.string_value={_EVALUATION_PIPELINE_COMPONENT_IDENTIFIER} AND metadata."input:model_name".string_value={model_name} AND (metadata."input:evaluation_task".string_value={_TEXT_GENERATION_TASK_NAME} OR metadata."input:evaluation_task".string_value={_SUMMARIZATION_TASK_NAME} OR metadata."input:evaluation_task".string_value={_QA_TASK_NAME} OR metadata."input:evaluation_task".string_value={_TEXT_CLASSIFICATION_TASK_NAME})'

        # NOTE: when task_name is appended to the filter the block of OR filters in `filters` above becomes a no-op
        if task_name:
            filters += f' AND metadata."input:evaluation_task".string_value={task_name}'

        filtered_pipeline_executions = aiplatform.Execution.list(
            filter=filters,
            project=aiplatform_initializer.global_config.project,
            location=aiplatform_initializer.global_config.location,
            credentials=aiplatform_initializer.global_config.credentials,
        )

        model_eval_metrics = []

        # TODO (b/285950380): improve performance of this method
        for pipeline_execution in filtered_pipeline_executions:
            if "pipeline_job_resource_name" not in pipeline_execution.metadata:
                continue

            pipeline_job_resource = aiplatform.PipelineJob.get(
                resource_name=pipeline_execution.metadata["pipeline_job_resource_name"]
            )
            eval_job_state = pipeline_job_resource._gca_resource.state

            if (
                eval_job_state
                != gca_pipeline_state.PipelineState.PIPELINE_STATE_SUCCEEDED
            ):
                continue

            metrics = None

            if only_summary_metrics:
                metrics = _get_metrics_from_pipeline_task_details(pipeline_job_resource)
            else:
                gcs_uri = _get_gcs_uri_from_pipeline_task_details(pipeline_job_resource)
                if gcs_uri:
                    metrics = _get_metrics_from_gcs_uri(gcs_uri)

            eval_metrics = ListEvaluationMetricsResponse(
                metrics=metrics,
                input_dataset_paths=list(
                    pipeline_execution.metadata["input:batch_predict_gcs_source_uris"]
                ),
                evaluation_task=pipeline_execution.metadata["input:evaluation_task"],
            )

            model_eval_metrics.append(eval_metrics)

        return model_eval_metrics
