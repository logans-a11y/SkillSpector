# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for per-slot model resolution in ``skillspector.constants``."""

from __future__ import annotations

import pytest

from skillspector.constants import _resolve_slot_model


class _FakeProviderWithPrefix:
    WIRE_MODEL_PREFIX = "google/"

    def resolve_model(self, slot: str = "default") -> str:
        return "provider-default"


class _FakeProviderWithoutPrefix:
    def resolve_model(self, slot: str = "default") -> str:
        return "provider-default"


class TestResolveSlotModel:
    def test_per_slot_override_strips_wire_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A per-slot override copied verbatim from the wire form (e.g. the
        # VertexAI endpoint's expected model parameter) must resolve to the
        # bare label, matching what provider.resolve_model() already does
        # for the shared SKILLSPECTOR_MODEL env var.
        monkeypatch.setenv("SKILLSPECTOR_MODEL_DEFAULT", "google/gemini-3.5-flash")
        assert _resolve_slot_model("default", _FakeProviderWithPrefix()) == "gemini-3.5-flash"

    def test_per_slot_override_without_prefix_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL_DEFAULT", "gemini-3.5-flash")
        assert _resolve_slot_model("default", _FakeProviderWithPrefix()) == "gemini-3.5-flash"

    def test_provider_without_wire_prefix_attribute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SKILLSPECTOR_MODEL_DEFAULT", "some-model")
        assert _resolve_slot_model("default", _FakeProviderWithoutPrefix()) == "some-model"

    def test_no_override_falls_back_to_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SKILLSPECTOR_MODEL_DEFAULT", raising=False)
        assert _resolve_slot_model("default", _FakeProviderWithPrefix()) == "provider-default"
