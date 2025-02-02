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

from vertexai.language_models._language_models import (
    _PreviewChatModel,
    _PreviewCodeChatModel,
    _PreviewCodeGenerationModel,
    _PreviewTextEmbeddingModel,
    _PreviewTextGenerationModel,
    ChatMessage,
    ChatModel,
    ChatSession,
    CodeChatSession,
    InputOutputTextPair,
    TextEmbedding,
    TextGenerationResponse,
)

ChatModel = _PreviewChatModel
CodeChatModel = _PreviewCodeChatModel
CodeGenerationModel = _PreviewCodeGenerationModel
TextGenerationModel = _PreviewTextGenerationModel
TextEmbeddingModel = _PreviewTextEmbeddingModel

__all__ = [
    "ChatMessage",
    "ChatModel",
    "ChatSession",
    "CodeChatModel",
    "CodeChatSession",
    "CodeGenerationModel",
    "InputOutputTextPair",
    "TextEmbedding",
    "TextEmbeddingModel",
    "TextGenerationModel",
    "TextGenerationResponse",
]
