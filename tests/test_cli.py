"""
CLI benchmark 入口測試
=====================

What:
- 驗證 `cfd-taichi benchmark` 能列出 registry 與執行正式 benchmark

Why:
- 若 CLI 仍只會轉發 examples，就還不算 toolkit-first 入口
"""

from __future__ import annotations

import numpy as np
import pytest
import sys

from cfd_taichi import cli


def test_cli_benchmark_list_prints_registry(tmp_path, capsys, monkeypatch):
    payload_path = tmp_path / "benchmark_index.npy"
    monkeypatch.setattr(
        sys,
        "argv",
        ["cfd-taichi", "benchmark", "list", "--payload-file", str(payload_path)],
    )

    cli.main()
    out = capsys.readouterr().out
    payload = np.load(payload_path, allow_pickle=True).item()

    assert "=== Benchmark Index ===" in out
    assert "name: lid_driven_cavity" in out
    assert "name: flow_over_cylinder" in out
    assert "name: cd_nozzle_euler" in out
    assert "name: naca0012_euler" in out
    assert "name: naca0012_ns" in out
    assert "name: poiseuille_flow_ns" in out
    assert "name: couette_flow_ns" in out
    assert "aliases:" in out
    assert "tags:" in out
    assert "default_params:" in out
    assert "acceptance_metrics:" in out
    assert "description:" in out
    assert "history-sampled[metric=history_samples;" in out
    assert "=== Payload File ===" in out
    assert payload["schema_name"] == "benchmark_index"
    assert payload["count"] >= 8
    assert "name" in payload["field_order"].tolist()
    assert "default_params" in payload["field_order"].tolist()
    assert "naca0012_euler" in payload["name"].tolist()
    assert "naca0012_ns" in payload["name"].tolist()
    assert "poiseuille_flow_ns" in payload["name"].tolist()
    assert "couette_flow_ns" in payload["name"].tolist()
    assert "cd_nozzle_euler" in payload["name"].tolist()
    assert "flow_over_cylinder" in payload["name"].tolist()


def test_cli_benchmark_run_executes_registry_runner(tmp_path, capsys, monkeypatch):
    payload_path = tmp_path / "summary.npy"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cfd-taichi",
            "benchmark",
            "poiseuille",
            "--arch",
            "cpu",
            "--steps",
            "2",
            "--sample-interval",
            "1",
            "--output-dir",
            str(tmp_path),
            "--save-state",
            "--save-history",
            "--payload-file",
            str(payload_path),
            "--set",
            "ni=24",
            "--set",
            "nj=12",
            "--set",
            "re=20.0",
            "--set",
            "u_max=0.05",
        ],
    )

    cli.main()
    out = capsys.readouterr().out
    payload = np.load(payload_path, allow_pickle=True).item()

    assert "=== Benchmark Summary ===" in out
    assert "benchmark: poiseuille_flow_ns" in out
    assert "requested_name: poiseuille_flow_ns" in out
    assert "solver_family: fvm" in out
    assert "equation_set: navier_stokes" in out
    assert "regime: incompressible" in out
    assert "acceptance_passed: True" in out
    assert "acceptance_failed_count: 0" in out
    assert "l2_profile_error:" in out
    assert "div_linf:" in out
    assert "=== Payload File ===" in out
    assert (tmp_path / "poiseuille_flow_ns" / "state_final.npy").exists()
    assert (tmp_path / "poiseuille_flow_ns" / "history.npy").exists()
    assert payload_path.exists()
    assert payload["schema_name"] == "benchmark_summary"
    assert payload["benchmark"] == "poiseuille_flow_ns"
    assert payload["requested_name"] == "poiseuille_flow_ns"
    assert payload["solver_family"] == "fvm"
    assert payload["equation_set"] == "navier_stokes"
    assert payload["regime"] == "incompressible"
    assert bool(payload["acceptance_passed"]) is True


def test_cli_rejects_legacy_case_dispatch(capsys, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["cfd-taichi", "cylinder"],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert "Unsupported command: cylinder" in out
    assert "cfd-taichi benchmark <name> [options]" in out
