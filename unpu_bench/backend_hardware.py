from __future__ import annotations

import glob
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .errors import CompilationError
from .muir import BackendArtifact

if TYPE_CHECKING:
    from .pipeline import CompileConfig


def _relative(path: Path, root: Path) -> str:
    return os.path.relpath(path, root)


def _run_cmd(cmd: list[str], *, cwd: Path) -> tuple[str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise CompilationError(
            "Backend toolchain command failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout, proc.stderr


def _find_newest(pattern: str) -> Path | None:
    matches = [Path(p) for p in glob.glob(pattern)]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _emit_tflm_passthrough(cfg: CompileConfig, out_root: Path) -> list[BackendArtifact]:
    if not cfg.backend_source_model:
        return []
    src = Path(cfg.backend_source_model)
    if src.suffix.lower() != ".tflite":
        raise CompilationError("tflm hardware artifact expects --backend-source-model to be a .tflite file.")
    out_dir = out_root / "tflm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.tflm.tflite"
    shutil.copy2(src, out_path)
    return [
        BackendArtifact(
            backend="tflm",
            artifact_type="hardware_model",
            path=_relative(out_path, out_root),
            meta={"format": "tflite", "optimized": False, "source": str(src)},
        )
    ]


def _emit_vela(cfg: CompileConfig, out_root: Path) -> list[BackendArtifact]:
    if not cfg.backend_source_model:
        return []
    src = Path(cfg.backend_source_model)
    if src.suffix.lower() != ".tflite":
        raise CompilationError("vela hardware artifact expects --backend-source-model to be a .tflite file.")
    vela = shutil.which("vela")
    if not vela:
        raise CompilationError(
            "Vela executable not found in PATH. Install Vela or disable hardware artifact emission."
        )

    out_dir = out_root / "vela"
    out_dir.mkdir(parents=True, exist_ok=True)
    accel = cfg.target_hardware
    if accel == "hxwe2":
        accel = "ethos-u55-64"
    cmd = [vela, "--accelerator-config", accel, str(src), "--output-dir", str(out_dir)]
    cmd += list(cfg.backend_tool_args or [])
    stdout, stderr = _run_cmd(cmd, cwd=out_root)
    (out_dir / "vela.log").write_text(f"{stdout}\n{stderr}", encoding="utf-8")

    cand = _find_newest(str(out_dir / "*.tflite"))
    if cand is None:
        raise CompilationError(
            f"Vela completed but no .tflite output was found in {out_dir}"
        )

    return [
        BackendArtifact(
            backend="vela",
            artifact_type="hardware_model",
            path=_relative(cand, out_root),
            meta={"format": "tflite", "optimized": True, "source": str(src), "tool": "vela"},
        )
    ]


def _cvi_quant_mode(bit_width: int | str) -> str:
    quant = {
        4: "INT4",
        8: "INT8",
        16: "F16",
        32: "F32",
        "F32": "F32",
        "BF16": "BF16",
        "F16": "F16",
        "INT8": "INT8",
        "INT4": "INT4",
        "W8F16": "W8F16",
        "W8BF16": "W8BF16",
        "W4F16": "W4F16",
        "W4BF16": "W4BF16",
        "F8E4M3": "F8E4M3",
        "F8E5M2": "F8E5M2",
        "QDQ": "QDQ",
    }
    if bit_width not in quant:
        raise CompilationError(f"Unsupported CVI bit-width/quant mode: {bit_width}")
    return quant[bit_width]


def _emit_cvi(cfg: CompileConfig, out_root: Path) -> list[BackendArtifact]:
    if not cfg.backend_source_model:
        return []
    src = Path(cfg.backend_source_model)
    if src.suffix.lower() != ".onnx":
        raise CompilationError("cvi hardware artifact expects --backend-source-model to be an .onnx file.")

    transform_bin = shutil.which("model_transform.py")
    cali_bin = shutil.which("run_calibration.py")
    deploy_bin = shutil.which("model_deploy.py")
    missing_bins = [n for n, p in [("model_transform.py", transform_bin), ("run_calibration.py", cali_bin), ("model_deploy.py", deploy_bin)] if p is None]
    if missing_bins:
        raise CompilationError(
            "CVI toolchain binaries not found in PATH: " + ", ".join(missing_bins)
        )

    out_dir = out_root / "cvi"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = src.stem
    model_mlir = out_dir / f"{model_name}.mlir"
    cal_table = Path(cfg.cvi_calibration_table) if cfg.cvi_calibration_table else (out_dir / f"{model_name}.table")
    output_model = out_dir / f"{model_name}.cvimodel"

    # 1) transform
    transform_cmd = [
        str(transform_bin),
        "--model_name",
        model_name,
        "--model_def",
        str(src.resolve()),
        "--mlir",
        str(model_mlir),
        "--output_names",
        cfg.output_names,
    ]
    if cfg.cvi_resize_dims:
        transform_cmd += ["--resize_dims", str(cfg.cvi_resize_dims)]
    if cfg.cvi_pixel_format:
        transform_cmd += ["--pixel_format", str(cfg.cvi_pixel_format)]
    if cfg.cvi_test_result:
        transform_cmd += ["--test_result", str(cfg.cvi_test_result)]
    if cfg.cvi_excepts:
        transform_cmd += ["--excepts", str(cfg.cvi_excepts)]
    if cfg.cvi_keep_aspect_ratio:
        transform_cmd += ["--keep_aspect_ratio"]
    transform_cmd += list(cfg.backend_tool_args or [])
    transform_stdout, transform_stderr = _run_cmd(transform_cmd, cwd=out_root)

    # 2) calibration (generate a tiny dataset folder from provided sample or random)
    cvi_data = out_dir / "cvi_data"
    cvi_data.mkdir(parents=True, exist_ok=True)
    ds_path = cvi_data / "ds_sample.npy"
    if cfg.data_sample and Path(cfg.data_sample).exists():
        arr = np.load(cfg.data_sample)
    else:
        # fallback sample from declared input shape
        shape = tuple(int(x) for x in cfg.input_shape.split())
        arr = np.random.randn(*shape).astype(np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    np.save(ds_path, arr.astype(np.int64))

    cali_cmd = [
        str(cali_bin),
        str(model_mlir),
        "--dataset",
        str(cvi_data),
        "--input_num",
        "1",
        "-o",
        str(cal_table),
    ]
    cali_stdout, cali_stderr = _run_cmd(cali_cmd, cwd=out_root)

    # 3) deploy
    deploy_cmd = [
        str(deploy_bin),
        "--mlir",
        str(model_mlir),
        "--quantize",
        _cvi_quant_mode(cfg.bit_width),
        "--calibration_table",
        str(cal_table),
        "--processor",
        cfg.target_hardware,
        "--tolerance",
        str(cfg.cvi_tolerance),
        "--model",
        str(output_model),
    ]
    if cfg.cvi_dynamic:
        deploy_cmd.append("--dynamic")
    if cfg.cvi_excepts:
        deploy_cmd += ["--excepts", str(cfg.cvi_excepts)]
    deploy_stdout, deploy_stderr = _run_cmd(deploy_cmd, cwd=out_root)

    (out_dir / "cvi.log").write_text(
        "\n".join(
            [
                "[model_transform]",
                transform_stdout,
                transform_stderr,
                "[run_calibration]",
                cali_stdout,
                cali_stderr,
                "[model_deploy]",
                deploy_stdout,
                deploy_stderr,
            ]
        ),
        encoding="utf-8",
    )

    if not output_model.exists():
        raise CompilationError(f"CVI pipeline finished but output model missing: {output_model}")

    return [
        BackendArtifact(
            backend="cvi",
            artifact_type="hardware_model",
            path=_relative(output_model, out_root),
            meta={
                "format": "cvimodel",
                "optimized": True,
                "source": str(src),
                "tool": "cvi_toolchain",
            },
        )
    ]


def _emit_eiq(cfg: CompileConfig, out_root: Path) -> list[BackendArtifact]:
    if not cfg.backend_source_model:
        return []
    src = Path(cfg.backend_source_model)
    if src.suffix.lower() != ".tflite":
        raise CompilationError("eiq hardware artifact expects --backend-source-model to be a .tflite file.")

    neutron = os.environ.get("EIQ_NEUTRON_PATH")
    if neutron:
        neutron_bin = Path(neutron)
    else:
        found = shutil.which("neutron")
        neutron_bin = Path(found) if found else Path()
    if not neutron_bin or not neutron_bin.exists():
        raise CompilationError(
            "eIQ compiler not found. Set EIQ_NEUTRON_PATH or add 'neutron' to PATH."
        )

    out_dir = out_root / "eiq"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_model = out_dir / f"{src.stem}_eiq.tflite"

    cmd = [
        str(neutron_bin),
        "--input",
        str(src),
        "--output",
        str(out_model),
    ]
    cmd += list(cfg.backend_tool_args or [])
    stdout, stderr = _run_cmd(cmd, cwd=out_root)
    (out_dir / "eiq.log").write_text(f"{stdout}\n{stderr}", encoding="utf-8")

    if not out_model.exists():
        raise CompilationError(f"eIQ compiler finished but output model missing: {out_model}")

    return [
        BackendArtifact(
            backend="eiq",
            artifact_type="hardware_model",
            path=_relative(out_model, out_root),
            meta={
                "format": "tflite",
                "optimized": True,
                "source": str(src),
                "tool": str(neutron_bin),
            },
        )
    ]


def _emit_external(cfg: CompileConfig, out_root: Path) -> list[BackendArtifact]:
    if not cfg.backend_command:
        return []
    if not cfg.backend_source_model:
        raise CompilationError("--backend-command requires --backend-source-model.")

    src = Path(cfg.backend_source_model)
    out_dir = out_root / cfg.target_format
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd_text = cfg.backend_command.format(
        input=str(src),
        out_dir=str(out_dir),
        backend=cfg.target_format,
    )
    cmd = shlex.split(cmd_text)
    stdout, stderr = _run_cmd(cmd, cwd=out_root)
    (out_dir / f"{cfg.target_format}.tool.log").write_text(f"{stdout}\n{stderr}", encoding="utf-8")

    pattern = cfg.backend_output_glob or "*"
    cand = _find_newest(str(out_dir / pattern))
    if cand is None:
        raise CompilationError(
            f"Backend command completed but no output matched '{pattern}' in {out_dir}"
        )
    return [
        BackendArtifact(
            backend=cfg.target_format,
            artifact_type="hardware_model",
            path=_relative(cand, out_root),
            meta={"format": cand.suffix.lstrip("."), "source": str(src), "command": cmd_text},
        )
    ]


def emit_hardware_artifacts(cfg: CompileConfig) -> list[BackendArtifact]:
    out_root = Path(cfg.out_dir)
    if cfg.target_format == "tflm":
        return _emit_tflm_passthrough(cfg, out_root)
    if cfg.target_format == "vela":
        return _emit_vela(cfg, out_root)
    if cfg.target_format == "cvi":
        return _emit_cvi(cfg, out_root)
    if cfg.target_format == "eiq":
        return _emit_eiq(cfg, out_root)
    # Any other external backend currently relies on command templates.
    return _emit_external(cfg, out_root)
