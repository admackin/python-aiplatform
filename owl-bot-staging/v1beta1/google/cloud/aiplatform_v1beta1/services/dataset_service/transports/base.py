# -*- coding: utf-8 -*-
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
import abc
from typing import Awaitable, Callable, Dict, Optional, Sequence, Union

from google.cloud.aiplatform_v1beta1 import gapic_version as package_version

import google.auth  # type: ignore
import google.api_core
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.api_core import operations_v1
from google.auth import credentials as ga_credentials  # type: ignore
from google.oauth2 import service_account # type: ignore

from google.cloud.aiplatform_v1beta1.types import annotation_spec
from google.cloud.aiplatform_v1beta1.types import dataset
from google.cloud.aiplatform_v1beta1.types import dataset as gca_dataset
from google.cloud.aiplatform_v1beta1.types import dataset_service
from google.cloud.location import locations_pb2 # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from google.longrunning import operations_pb2
from google.longrunning import operations_pb2  # type: ignore

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(gapic_version=package_version.__version__)


class DatasetServiceTransport(abc.ABC):
    """Abstract transport class for DatasetService."""

    AUTH_SCOPES = (
        'https://www.googleapis.com/auth/cloud-platform',
    )

    DEFAULT_HOST: str = 'aiplatform.googleapis.com'
    def __init__(
            self, *,
            host: str = DEFAULT_HOST,
            credentials: Optional[ga_credentials.Credentials] = None,
            credentials_file: Optional[str] = None,
            scopes: Optional[Sequence[str]] = None,
            quota_project_id: Optional[str] = None,
            client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
            always_use_jwt_access: Optional[bool] = False,
            api_audience: Optional[str] = None,
            **kwargs,
            ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to.
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is mutually exclusive with credentials.
            scopes (Optional[Sequence[str]]): A list of scopes.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
        """

        scopes_kwargs = {"scopes": scopes, "default_scopes": self.AUTH_SCOPES}

        # Save the scopes.
        self._scopes = scopes

        # If no credentials are provided, then determine the appropriate
        # defaults.
        if credentials and credentials_file:
            raise core_exceptions.DuplicateCredentialArgs("'credentials_file' and 'credentials' are mutually exclusive")

        if credentials_file is not None:
            credentials, _ = google.auth.load_credentials_from_file(
                                credentials_file,
                                **scopes_kwargs,
                                quota_project_id=quota_project_id
                            )
        elif credentials is None:
            credentials, _ = google.auth.default(**scopes_kwargs, quota_project_id=quota_project_id)
            # Don't apply audience if the credentials file passed from user.
            if hasattr(credentials, "with_gdch_audience"):
                credentials = credentials.with_gdch_audience(api_audience if api_audience else host)

        # If the credentials are service account credentials, then always try to use self signed JWT.
        if always_use_jwt_access and isinstance(credentials, service_account.Credentials) and hasattr(service_account.Credentials, "with_always_use_jwt_access"):
            credentials = credentials.with_always_use_jwt_access(True)

        # Save the credentials.
        self._credentials = credentials

        # Save the hostname. Default to port 443 (HTTPS) if none is specified.
        if ':' not in host:
            host += ':443'
        self._host = host

    def _prep_wrapped_messages(self, client_info):
        # Precompute the wrapped methods.
        self._wrapped_methods = {
            self.create_dataset: gapic_v1.method.wrap_method(
                self.create_dataset,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.get_dataset: gapic_v1.method.wrap_method(
                self.get_dataset,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.update_dataset: gapic_v1.method.wrap_method(
                self.update_dataset,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.list_datasets: gapic_v1.method.wrap_method(
                self.list_datasets,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.delete_dataset: gapic_v1.method.wrap_method(
                self.delete_dataset,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.import_data: gapic_v1.method.wrap_method(
                self.import_data,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.export_data: gapic_v1.method.wrap_method(
                self.export_data,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.list_data_items: gapic_v1.method.wrap_method(
                self.list_data_items,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.search_data_items: gapic_v1.method.wrap_method(
                self.search_data_items,
                default_timeout=None,
                client_info=client_info,
            ),
            self.list_saved_queries: gapic_v1.method.wrap_method(
                self.list_saved_queries,
                default_timeout=None,
                client_info=client_info,
            ),
            self.delete_saved_query: gapic_v1.method.wrap_method(
                self.delete_saved_query,
                default_timeout=None,
                client_info=client_info,
            ),
            self.get_annotation_spec: gapic_v1.method.wrap_method(
                self.get_annotation_spec,
                default_timeout=5.0,
                client_info=client_info,
            ),
            self.list_annotations: gapic_v1.method.wrap_method(
                self.list_annotations,
                default_timeout=5.0,
                client_info=client_info,
            ),
         }

    def close(self):
        """Closes resources associated with the transport.

       .. warning::
            Only call this method if the transport is NOT shared
            with other clients - this may cause errors in other clients!
        """
        raise NotImplementedError()

    @property
    def operations_client(self):
        """Return the client designed to process long-running operations."""
        raise NotImplementedError()

    @property
    def create_dataset(self) -> Callable[
            [dataset_service.CreateDatasetRequest],
            Union[
                operations_pb2.Operation,
                Awaitable[operations_pb2.Operation]
            ]]:
        raise NotImplementedError()

    @property
    def get_dataset(self) -> Callable[
            [dataset_service.GetDatasetRequest],
            Union[
                dataset.Dataset,
                Awaitable[dataset.Dataset]
            ]]:
        raise NotImplementedError()

    @property
    def update_dataset(self) -> Callable[
            [dataset_service.UpdateDatasetRequest],
            Union[
                gca_dataset.Dataset,
                Awaitable[gca_dataset.Dataset]
            ]]:
        raise NotImplementedError()

    @property
    def list_datasets(self) -> Callable[
            [dataset_service.ListDatasetsRequest],
            Union[
                dataset_service.ListDatasetsResponse,
                Awaitable[dataset_service.ListDatasetsResponse]
            ]]:
        raise NotImplementedError()

    @property
    def delete_dataset(self) -> Callable[
            [dataset_service.DeleteDatasetRequest],
            Union[
                operations_pb2.Operation,
                Awaitable[operations_pb2.Operation]
            ]]:
        raise NotImplementedError()

    @property
    def import_data(self) -> Callable[
            [dataset_service.ImportDataRequest],
            Union[
                operations_pb2.Operation,
                Awaitable[operations_pb2.Operation]
            ]]:
        raise NotImplementedError()

    @property
    def export_data(self) -> Callable[
            [dataset_service.ExportDataRequest],
            Union[
                operations_pb2.Operation,
                Awaitable[operations_pb2.Operation]
            ]]:
        raise NotImplementedError()

    @property
    def list_data_items(self) -> Callable[
            [dataset_service.ListDataItemsRequest],
            Union[
                dataset_service.ListDataItemsResponse,
                Awaitable[dataset_service.ListDataItemsResponse]
            ]]:
        raise NotImplementedError()

    @property
    def search_data_items(self) -> Callable[
            [dataset_service.SearchDataItemsRequest],
            Union[
                dataset_service.SearchDataItemsResponse,
                Awaitable[dataset_service.SearchDataItemsResponse]
            ]]:
        raise NotImplementedError()

    @property
    def list_saved_queries(self) -> Callable[
            [dataset_service.ListSavedQueriesRequest],
            Union[
                dataset_service.ListSavedQueriesResponse,
                Awaitable[dataset_service.ListSavedQueriesResponse]
            ]]:
        raise NotImplementedError()

    @property
    def delete_saved_query(self) -> Callable[
            [dataset_service.DeleteSavedQueryRequest],
            Union[
                operations_pb2.Operation,
                Awaitable[operations_pb2.Operation]
            ]]:
        raise NotImplementedError()

    @property
    def get_annotation_spec(self) -> Callable[
            [dataset_service.GetAnnotationSpecRequest],
            Union[
                annotation_spec.AnnotationSpec,
                Awaitable[annotation_spec.AnnotationSpec]
            ]]:
        raise NotImplementedError()

    @property
    def list_annotations(self) -> Callable[
            [dataset_service.ListAnnotationsRequest],
            Union[
                dataset_service.ListAnnotationsResponse,
                Awaitable[dataset_service.ListAnnotationsResponse]
            ]]:
        raise NotImplementedError()

    @property
    def list_operations(
        self,
    ) -> Callable[
        [operations_pb2.ListOperationsRequest],
        Union[operations_pb2.ListOperationsResponse, Awaitable[operations_pb2.ListOperationsResponse]],
    ]:
        raise NotImplementedError()

    @property
    def get_operation(
        self,
    ) -> Callable[
        [operations_pb2.GetOperationRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def cancel_operation(
        self,
    ) -> Callable[
        [operations_pb2.CancelOperationRequest],
        None,
    ]:
        raise NotImplementedError()

    @property
    def delete_operation(
        self,
    ) -> Callable[
        [operations_pb2.DeleteOperationRequest],
        None,
    ]:
        raise NotImplementedError()

    @property
    def wait_operation(
        self,
    ) -> Callable[
        [operations_pb2.WaitOperationRequest],
        Union[operations_pb2.Operation, Awaitable[operations_pb2.Operation]],
    ]:
        raise NotImplementedError()

    @property
    def set_iam_policy(
        self,
    ) -> Callable[
        [iam_policy_pb2.SetIamPolicyRequest],
        Union[policy_pb2.Policy, Awaitable[policy_pb2.Policy]],
    ]:
        raise NotImplementedError()

    @property
    def get_iam_policy(
        self,
    ) -> Callable[
        [iam_policy_pb2.GetIamPolicyRequest],
        Union[policy_pb2.Policy, Awaitable[policy_pb2.Policy]],
    ]:
        raise NotImplementedError()

    @property
    def test_iam_permissions(
        self,
    ) -> Callable[
        [iam_policy_pb2.TestIamPermissionsRequest],
        Union[
            iam_policy_pb2.TestIamPermissionsResponse,
            Awaitable[iam_policy_pb2.TestIamPermissionsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_location(self,
    ) -> Callable[
        [locations_pb2.GetLocationRequest],
        Union[locations_pb2.Location, Awaitable[locations_pb2.Location]],
    ]:
        raise NotImplementedError()

    @property
    def list_locations(self,
    ) -> Callable[
        [locations_pb2.ListLocationsRequest],
        Union[locations_pb2.ListLocationsResponse, Awaitable[locations_pb2.ListLocationsResponse]],
    ]:
        raise NotImplementedError()

    @property
    def kind(self) -> str:
        raise NotImplementedError()


__all__ = (
    'DatasetServiceTransport',
)
