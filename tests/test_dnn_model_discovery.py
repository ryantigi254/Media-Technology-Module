from __future__ import annotations

from pathlib import Path

from ssc.features.dnn_person_detector import find_default_dnn_model_path


def test_find_default_dnn_model_path_returns_proto(tmp_path: Path) -> None:
    proto = tmp_path / "model.prototxt"
    model = tmp_path / "model.caffemodel"
    proto.write_text("proto", encoding="utf-8")
    model.write_text("model", encoding="utf-8")

    found = find_default_dnn_model_path(tmp_path)

    assert found == proto


def test_find_default_dnn_model_path_none(tmp_path: Path) -> None:
    found = find_default_dnn_model_path(tmp_path)

    assert found is None
