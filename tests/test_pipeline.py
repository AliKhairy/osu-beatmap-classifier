"""
Tests for the wiring of the Prefect flow.

The flow's failure here was silent until a real run: it defaulted to v1
features, and its export step never lifted export-onnx's v2 safeguard, so a v2
candidate was trained and promoted, and then the export refused it. The models
in the root then no longer matched the ONNX files the app ships. These tests pin
both halves without training anything.
"""
import logging

import pytest

pytest.importorskip('prefect')

import cli  # noqa: E402
from flows import pipeline  # noqa: E402


def test_pipeline_trains_v2_by_default():
    assert cli.build_parser().parse_args(['pipeline']).feature_version == 2


@pytest.mark.parametrize('version', [1, 2])
def test_export_allows_exactly_the_trained_version(monkeypatch, tmp_path, version):
    seen = {}

    def fake_export(args):
        seen.update(vars(args))
        return 0

    monkeypatch.setattr(pipeline, 'get_run_logger', lambda: logging.getLogger('test'))
    monkeypatch.setattr(cli, 'cmd_export_onnx', fake_export)
    pipeline.export_task.fn(str(tmp_path), version)

    assert seen['allow_feature_version'] == version
    assert seen['model_dir'] == str(tmp_path)


def test_export_still_refuses_a_version_it_was_not_given(tmp_path):
    # The safeguard itself: a v2 model with no permission must not export.
    (tmp_path / 'ensemble_model_1.keras').write_bytes(b'')
    (tmp_path / 'feature_meta.json').write_text('{"feature_version": 2}')
    args = cli.argparse.Namespace(model_dir=str(tmp_path), out_dir=str(tmp_path),
                                  allow_feature_version=None)
    assert cli.cmd_export_onnx(args) == 1
