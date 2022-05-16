# -*- coding: utf-8 -*-

# Copyright 2021 Google LLC
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
import numpy as np
import pandas as pd
import pytest

from google.api_core import exceptions

from google.cloud import aiplatform
from tests.system.aiplatform import e2e_base

_RUN = 'run-1'
_PARAMS = {"sdk-param-test-1": 0.1, "sdk-param-test-2": 0.2}
_METRICS = {"sdk-metric-test-1": 0.8, "sdk-metric-test-2": 100.0}

_RUN_2 = 'run-2'
_PARAMS_2 = {"sdk-param-test-1": 0.2, "sdk-param-test-2": 0.4}
_METRICS_2 = {"sdk-metric-test-1": 1.6, "sdk-metric-test-2": 200.0}

_TIME_SERIES_METRIC_KEY = "accuracy"

_URI = 'test-uri'


class TestExperiments(e2e_base.TestEndToEnd):

    _temp_prefix = "tmpvrtxsdk-e2e"

    def setup_class(cls):
        cls._experiment_name = cls._make_display_name("experiment")[:30]
        cls._dataset_artifact_name = cls._make_display_name('ds-artifact')[:30]

    def test_create_experiment(self, shared_state):

        # Truncating the name because of resource id constraints from the service

        tensorboard = aiplatform.Tensorboard.create(
            project=e2e_base._PROJECT,
            location=e2e_base._LOCATION,
            display_name=self._experiment_name)

        shared_state['resources'] = [tensorboard]

        aiplatform.init(
            project=e2e_base._PROJECT,
            location=e2e_base._LOCATION,
            experiment=self._experiment_name,
            experiment_tensorboard=tensorboard
        )

        shared_state["resources"].append(aiplatform.metadata.experiment_tracker.experiment)

    def test_get_experiment(self):
        experiment = aiplatform.Experiment(experiment_name=self._experiment_name)
        assert experiment.name == self._experiment_name

    def test_start_run(self):
        run = aiplatform.start_run(_RUN)
        assert run.name == _RUN

    def test_get_run(self):
        run = aiplatform.ExperimentRun(
            run_name=_RUN,
            experiment=self._experiment_name)
        assert run.name == _RUN
        assert run.state == aiplatform.gapic.Execution.State.RUNNING

    def test_log_params(self):
        aiplatform.log_params(_PARAMS)
        run = aiplatform.ExperimentRun(
            run_name=_RUN,
            experiment=self._experiment_name)
        assert run.get_params() == _PARAMS

    def test_log_metrics(self):
        aiplatform.log_metrics(_METRICS)
        run = aiplatform.ExperimentRun(
            run_name=_RUN,
            experiment=self._experiment_name)
        assert run.get_metrics() == _METRICS

    def test_log_time_series_metrics(self):
        for i in range(5):
            aiplatform.log_time_series_metrics({_TIME_SERIES_METRIC_KEY: i})

        run = aiplatform.ExperimentRun(
            run_name=_RUN,
            experiment=self._experiment_name)

        time_series_result = run.get_time_series_dataframe()[[_TIME_SERIES_METRIC_KEY, 'step']].to_dict('list')

        assert time_series_result == {'step': list(range(1, 6)),
                                      _TIME_SERIES_METRIC_KEY: [float(value) for value in range(5)]}

    def test_create_artifact(self, shared_state):
        ds = aiplatform.Artifact.create(
            schema_title='system.Dataset',
            resource_id=self._dataset_artifact_name,
            uri=_URI)

        shared_state['resources'].append(ds)
        assert ds.uri == _URI

    def test_log_execution_and_artifact(self, shared_state):
        with aiplatform.start_execution(
                schema_title='system.ContainerExecution',
                resource_id=self._make_display_name('execution')) as execution:

            shared_state['resources'].append(execution)

            ds = aiplatform.Artifact(resource_name=self._dataset_artifact_name)
            execution.assign_input_artifacts([ds])

            model = aiplatform.Artifact.create(schema_title='system.Model')
            execution.assign_output_artifacts([model])
            shared_state['resources'].append(model)

        input_artifacts = execution.get_input_artifacts()
        assert input_artifacts[0].name == ds.name

        output_artifacts = execution.get_output_artifacts()
        assert output_artifacts[0].name == model.name

        run = aiplatform.ExperimentRun(run_name=_RUN, experiment=self._experiment_name)
        executions = run.get_executions()
        assert executions[0].name == execution.name

        artifacts = run.get_artifacts()
        #tensorboard run artifact is also included
        assert sorted([artifact.name for artifact in artifacts]) == sorted(
            [ds.name, model.name, run._tensorboard_run_id(run.resource_id)])

    def test_end_run(self):
        aiplatform.end_run()
        run = aiplatform.ExperimentRun(
            run_name=_RUN,
            experiment=self._experiment_name)
        assert run.state == aiplatform.gapic.Execution.State.COMPLETE

    def test_run_context_manager(self):
        with aiplatform.start_run(_RUN_2) as run:
            run.log_params(_PARAMS_2)
            run.log_metrics(_METRICS_2)
            assert run.state == aiplatform.gapic.Execution.State.RUNNING

        assert run.state == aiplatform.gapic.Execution.State.COMPLETE

    def test_get_experiments_df(self):
        df = aiplatform.get_experiment_df()

        true_df_dict_1 = {f"metric.{key}": value for key, value in _METRICS.items()}
        for key, value in _PARAMS.items():
            true_df_dict_1[f"param.{key}"] = value

        true_df_dict_1["experiment_name"] = self._experiment_name
        true_df_dict_1["run_name"] = _RUN
        true_df_dict_1["state"] = aiplatform.gapic.Execution.State.COMPLETE.name
        true_df_dict_1["run_type"] = aiplatform.metadata.constants.SYSTEM_EXPERIMENT_RUN
        true_df_dict_1[f"time_series_metric.{_TIME_SERIES_METRIC_KEY}"] = 4.0

        true_df_dict_2 = {f"metric.{key}": value for key, value in _METRICS_2.items()}
        for key, value in _PARAMS_2.items():
            true_df_dict_2[f"param.{key}"] = value

        true_df_dict_2["experiment_name"] = self._experiment_name
        true_df_dict_2["run_name"] = _RUN_2
        true_df_dict_2["state"] = aiplatform.gapic.Execution.State.COMPLETE.name
        true_df_dict_2["run_type"] = aiplatform.metadata.constants.SYSTEM_EXPERIMENT_RUN
        true_df_dict_2[f"time_series_metric.{_TIME_SERIES_METRIC_KEY}"] = float('nan')

        assert sorted(
            [true_df_dict_1, true_df_dict_2], key=lambda d: d["run_name"]
        ) == sorted(df.to_dict("records"), key=lambda d: d["run_name"])

    def test_delete_run(self):
        run = aiplatform.ExperimentRun(run_name=_RUN, experiment=self._experiment_name)
        run.delete(delete_backing_tensorboard_run=True)

        with pytest.raises(exceptions.NotFound):
            aiplatform.ExperimentRun(run_name=_RUN, experiment=self._experiment_name)

    def test_delete_experiment(self):
        experiment = aiplatform.Experiment(experiment_name=self._experiment_name)
        experiment.delete(delete_backing_tensorboard_runs=True)

        with pytest.raises(exceptions.NotFound):
            aiplatform.Experiment(experiment_name=self._experiment_name)