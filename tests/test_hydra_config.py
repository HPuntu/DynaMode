from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from dynamode.config import (
    config_fingerprint,
    derive_run_descriptors,
    flatten_sections,
    load_run_manifest,
    prepare_runtime_identity,
    prepare_training_config,
    split_inference_config,
    update_model_spec,
    write_run_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HYDRA_CONFIG_DIR = REPO_ROOT / "configs" / "hydra"


def _minimal_nested_config() -> dict:
    return {
        "core": {
            "checkpoint_dir": "checkpoints/test",
            "model_type": "spectral_conv_block_mix_amplitude",
            "run_name": "test",
        },
        "representation": {
            "representation": "displacement",
            "coords_type": "ca",
            "window_size": 256,
            "top_k_freqs": 256,
            "freq_normalization": "auto",
            "dc_residualization": "auto",
            "aniso_source": "auto",
        },
        "model": {
            "prediction_target": "x_0",
            "spectral_modes": 256,
        },
        "diffusion": {"num_steps": 1000},
        "sampling": {"num_ode_steps": 50},
        "training": {"geo_loss": "spec_geo"},
        "inference": {"input": None, "batch_size": None},
    }


def test_base_ca_hydra_composition_has_expected_scientific_setup():
    hydra = pytest.importorskip("hydra")

    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(HYDRA_CONFIG_DIR)):
        composed = hydra.compose(config_name="dynamode_conf")

    runtime = prepare_training_config(composed, resolve_identity=False)
    assert runtime["model_type"] == "spectral_conv_block_mix_amplitude"
    assert runtime["representation"] == "displacement"
    assert runtime["coords_type"] == "ca"
    assert runtime["top_k_freqs"] == 256
    assert runtime["spectral_modes"] == 256
    assert runtime["prediction_target"] == "x_0"
    assert runtime["geo_loss"] == "spec_geo"


def test_hydra_leaf_overrides_reach_flat_runtime_config():
    hydra = pytest.importorskip("hydra")

    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(HYDRA_CONFIG_DIR)):
        composed = hydra.compose(
            config_name="dynamode_conf",
            overrides=["diffusion.shift_value=0.5", "training.geometry_lambda=0.2"],
        )

    flat = flatten_sections(composed)
    assert flat["shift_value"] == 0.5
    assert flat["geometry_lambda"] == 0.2


def test_hydra_group_choices_become_automatic_handles():
    nested = _minimal_nested_config()
    prepared = prepare_training_config(
        nested,
        config_choices={
            "experiment": "base_ca",
            "model": "spec_conv_base",
            "hydra/job_logging": "disabled",
        },
        resolve_identity=False,
    )

    assert prepared["experiment_handle"] == "base_ca"
    assert prepared["model_handle"] == "spec_conv_base"
    assert prepared["config_choices"] == {
        "experiment": "base_ca",
        "model": "spec_conv_base",
    }


def test_flatten_sections_rejects_colliding_runtime_keys():
    with pytest.raises(ValueError, match="defined in both"):
        flatten_sections(
            {
                "core": {"window_size": 256},
                "representation": {"window_size": 128},
            }
        )


def test_flatten_sections_rejects_unnamed_top_level_values():
    with pytest.raises(ValueError, match="Unexpected top-level"):
        flatten_sections({"core": {"epochs": 1}, "stray": 2})


def test_prepare_training_config_preserves_resume_directory(tmp_path):
    nested = _minimal_nested_config()
    checkpoint_dir = tmp_path / "test"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint_latest.pt").touch()
    nested["core"]["checkpoint_dir"] = str(checkpoint_dir)
    nested["core"]["resume_from_latest"] = True
    nested["run"] = {"resume_config_mode": "current"}

    prepared = prepare_training_config(
        nested,
        now=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc),
    )

    assert prepared["checkpoint_dir"] == str(checkpoint_dir)
    assert prepared["run_name"] == prepared["run_id"]
    assert prepared["displacement"] is True


def test_automatic_run_identity_is_descriptive_and_date_partitioned(tmp_path):
    config = {
        "output_root": str(tmp_path),
        "checkpoint_dir": None,
        "experiment_handle": "base_ca",
        "model_handle": "specconv_base",
        "setup_tags": ["unit-var"],
        "model_type": "spectral_conv_block_mix_amplitude",
        "representation": "displacement",
        "coords_type": "ca",
        "use_DCT": True,
        "top_k_freqs": 256,
    }
    now = datetime(
        2026,
        8,
        20,
        14,
        32,
        18,
        123000,
        tzinfo=timezone(timedelta(hours=1)),
    )

    prepared = prepare_runtime_identity(config, now=now)

    assert prepared["experiment_handle"] == "base-ca"
    assert prepared["model_handle"] == "specconv-base"
    assert prepared["setup_handle"] == "ca-disp-dct-k256-unit-var"
    assert prepared["run_descriptor"] == (
        "base-ca__specconv-base__ca-disp-dct-k256-unit-var"
    )
    assert prepared["run_id"].startswith(
        "14-32-18__base-ca__specconv-base__ca-disp-dct-k256-unit-var__"
    )
    assert Path(prepared["checkpoint_dir"]).parent == tmp_path / "2026-08-20"
    assert prepared["run_name"] == prepared["run_id"]


def test_descriptors_and_fingerprint_follow_the_resolved_setup():
    base = {
        "experiment_handle": "base_ca",
        "model_type": "spectral_conv_block_mix_amplitude",
        "representation": "displacement",
        "coords_type": "bb",
        "use_DCT": True,
        "top_k_freqs": 128,
    }
    descriptors = derive_run_descriptors(base)

    assert descriptors["model_handle"] == "specconv-base"
    assert descriptors["setup_handle"] == "bb-disp-dct-k128"
    assert config_fingerprint(base) != config_fingerprint({**base, "top_k_freqs": 256})


def test_resume_recovers_the_existing_run_identity(tmp_path):
    base = {
        "output_root": str(tmp_path),
        "checkpoint_dir": None,
        "experiment_handle": "base_ca",
        "model_handle": "specconv_base",
        "model_type": "spectral_conv_block_mix_amplitude",
        "representation": "displacement",
        "coords_type": "ca",
        "use_DCT": True,
        "top_k_freqs": 256,
    }
    created = prepare_runtime_identity(
        base,
        now=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc),
    )
    write_run_metadata(created["checkpoint_dir"], _minimal_nested_config(), created)

    resumed = prepare_runtime_identity(
        {
            **base,
            "checkpoint_dir": created["checkpoint_dir"],
            "resume_from_latest": True,
        },
        now=datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc),
    )

    assert resumed["checkpoint_dir"] == created["checkpoint_dir"]
    assert resumed["run_id"] == created["run_id"]
    assert resumed["started_at_utc"] == created["started_at_utc"]


def test_resume_restores_recorded_config_and_allows_operational_overrides(tmp_path):
    run_dir = tmp_path / "recorded-run"
    run_dir.mkdir()
    (run_dir / "checkpoint_latest.pt").touch()
    recorded_nested = _minimal_nested_config()
    recorded_nested["core"].update(
        {
            "checkpoint_dir": str(run_dir),
            "epochs": 300,
        }
    )
    recorded_nested["model"]["spectral_modes"] = 128
    recorded = prepare_training_config(recorded_nested)
    write_run_metadata(run_dir, recorded_nested, recorded)
    update_model_spec(run_dir, {"model_class": "tests.RecordedModel"})

    current = _minimal_nested_config()
    current["core"].update(
        {
            "checkpoint_dir": str(run_dir),
            "resume_from_latest": True,
            "epochs": 450,
        }
    )
    current["model"]["spectral_modes"] = 256
    restored = prepare_training_config(
        current,
        task_overrides=[
            f"core.checkpoint_dir={run_dir}",
            "core.resume_from_latest=true",
            "core.epochs=450",
        ],
        resolve_identity=False,
    )

    assert restored["spectral_modes"] == 128
    assert restored["epochs"] == 450
    assert restored["checkpoint_dir"] == str(run_dir)
    assert restored["resume_from_latest"] is True

    write_run_metadata(
        run_dir,
        current,
        restored,
        overrides=["core.resume_from_latest=true", "core.epochs=450"],
    )
    manifest = yaml.safe_load((run_dir / "run_manifest.yaml").read_text())
    model_spec = yaml.safe_load((run_dir / "model_spec.yaml").read_text())
    assert len(manifest["resume_events"]) == 1
    assert manifest["configuration"]["composed"]["model"]["spectral_modes"] == 128
    assert model_spec["runtime"]["model_class"] == "tests.RecordedModel"


def test_resume_rejects_unapproved_scientific_overrides(tmp_path):
    run_dir = tmp_path / "recorded-run"
    run_dir.mkdir()
    (run_dir / "checkpoint_latest.pt").touch()
    recorded_nested = _minimal_nested_config()
    recorded_nested["core"]["checkpoint_dir"] = str(run_dir)
    recorded_nested["model"]["spectral_modes"] = 128
    recorded = prepare_training_config(recorded_nested)
    write_run_metadata(run_dir, recorded_nested, recorded)

    current = _minimal_nested_config()
    current["core"].update(
        {"checkpoint_dir": str(run_dir), "resume_from_latest": True}
    )
    current["model"]["spectral_modes"] = 256

    with pytest.raises(ValueError, match="spectral_modes"):
        prepare_training_config(
            current,
            task_overrides=[
                f"core.checkpoint_dir={run_dir}",
                "core.resume_from_latest=true",
                "model.spectral_modes=256",
            ],
            resolve_identity=False,
        )


def test_resume_requires_latest_checkpoint_even_with_manifest(tmp_path):
    run_dir = tmp_path / "recorded-run"
    recorded_nested = _minimal_nested_config()
    recorded_nested["core"]["checkpoint_dir"] = str(run_dir)
    recorded = prepare_training_config(recorded_nested)
    write_run_metadata(run_dir, recorded_nested, recorded)

    current = _minimal_nested_config()
    current["core"].update(
        {"checkpoint_dir": str(run_dir), "resume_from_latest": True}
    )

    with pytest.raises(FileNotFoundError, match="checkpoint_latest.pt"):
        prepare_training_config(current, resolve_identity=False)


def test_resume_does_not_fall_back_to_run_config_yaml(tmp_path):
    run_dir = tmp_path / "recorded-run"
    run_dir.mkdir()
    (run_dir / "checkpoint_latest.pt").touch()
    (run_dir / "run_config.yaml").write_text(
        yaml.safe_dump({"window_size": 256}),
        encoding="utf-8",
    )
    current = _minimal_nested_config()
    current["core"].update(
        {"checkpoint_dir": str(run_dir), "resume_from_latest": True}
    )

    with pytest.raises(FileNotFoundError, match="run_manifest.yaml"):
        prepare_training_config(current, resolve_identity=False)


def test_resume_ignores_inference_only_overrides(tmp_path):
    run_dir = tmp_path / "recorded-run"
    run_dir.mkdir()
    (run_dir / "checkpoint_latest.pt").touch()
    recorded_nested = _minimal_nested_config()
    recorded_nested["core"]["checkpoint_dir"] = str(run_dir)
    recorded_nested["data"] = {"batch_size": 200}
    recorded = prepare_training_config(recorded_nested)
    write_run_metadata(run_dir, recorded_nested, recorded)

    current = _minimal_nested_config()
    current["core"].update(
        {"checkpoint_dir": str(run_dir), "resume_from_latest": True}
    )
    current["data"] = {"batch_size": 64}
    restored = prepare_training_config(
        current,
        task_overrides=[
            f"core.checkpoint_dir={run_dir}",
            "core.resume_from_latest=true",
            "inference.batch_size=32",
        ],
        resolve_identity=False,
    )

    assert restored["batch_size"] == 200


def test_split_inference_config_applies_only_non_null_runtime_overrides():
    nested = _minimal_nested_config()
    nested["data"] = {"batch_size": 200, "crop_size": 576}
    nested["sampling"]["guidance_scale"] = 1.0
    nested["inference"].update(
        {
            "input": "target.pdb",
            "batch_size": 4,
            "crop_size": None,
            "guidance_scale": 1.5,
        }
    )

    runtime, inference = split_inference_config(nested)

    assert runtime["batch_size"] == 4
    assert runtime["crop_size"] == 576
    assert runtime["guidance_scale"] == 1.5
    assert inference["input"] == "target.pdb"


def test_load_run_manifest_rejects_sectioned_and_flat_configs(tmp_path):
    sectioned = tmp_path / "sectioned.yaml"
    flat = tmp_path / "flat.yaml"
    sectioned.write_text(yaml.safe_dump(_minimal_nested_config()), encoding="utf-8")
    flat.write_text(yaml.safe_dump({"window_size": 256}), encoding="utf-8")

    for path in (sectioned, flat):
        with pytest.raises(ValueError, match="run_manifest.yaml"):
            load_run_manifest(path)


def test_write_run_metadata_factorises_config_and_runtime_overlay(tmp_path):
    nested = _minimal_nested_config()
    flat = flatten_sections(nested)

    flat["run_id"] = "12-00-00__test__model__ca-disp-dct-k256__1234567890"
    write_run_metadata(tmp_path, nested, flat, ["model.spectral_modes=128"])

    manifest = yaml.safe_load((tmp_path / "run_manifest.yaml").read_text())
    assert manifest["configuration"]["composed"] == nested
    assert manifest["configuration"]["overrides"] == ["model.spectral_modes=128"]
    assert manifest["configuration"]["runtime_overlay"]["run_id"] == flat["run_id"]
    assert load_run_manifest(tmp_path / "run_manifest.yaml") == flat
    assert (tmp_path / "model_spec.yaml").exists()
    assert (tmp_path / "RUN.md").exists()
    assert not (tmp_path / "config_resolved.yaml").exists()
    assert not (tmp_path / "config_overrides.yaml").exists()
    assert not (tmp_path / "run_config.yaml").exists()
    assert not (tmp_path / "provenance.yaml").exists()

    update_model_spec(
        tmp_path,
        {
            "model_class": "tests.DummyModel",
            "parameters": {"total": 100, "trainable": 80},
        },
    )
    model_spec = yaml.safe_load((tmp_path / "model_spec.yaml").read_text())
    assert model_spec["runtime"]["model_class"] == "tests.DummyModel"
    assert model_spec["runtime"]["parameters"]["trainable"] == 80
