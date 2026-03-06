# uNPU-Bench

Unified IR and compilation framework for micro-NPU targets.

This repository provides:
- A backend-agnostic unified IR (`muir.py`)
- Canonicalization + validation passes over IR
- Capability-schema-driven legality and partitioning
- Multi-backend lowering (`tflm`, `vela`, `cvi`, `eiq`, `ai8x`)
- Optional hardware-native artifact emission via vendor toolchains

## Quick Start

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### 2. Run tests

```bash
./scripts/run_pytest.sh
```

## Compiler Usage

Basic compile:

```bash
unpu-bench \
  --target-format vela \
  --target-hardware hxwe2 \
  --bit-width 8 \
  --model-py models/my_model.py \
  --model-class MyModel \
  --input-shape "1 3 32 32" \
  --output-shape "1 10" \
  --input-names input \
  --output-names output \
  --out-dir out \
  --overwrite
```

Model source options (exactly one required):
- Torch module: `--model-py ... --model-class ...` (plus input/output metadata)
- ONNX file: `--model-onnx path/to/model.onnx`
- TFLite file: `--model-tflite path/to/model.tflite`

Outputs include:
- `program.json` (full compile metadata)
- `*/model.<backend>.ir.json` (backend IR bundle)
- `*/model.<backend>.compiled.json` (IR-native executable artifact for `unpu_ir_runtime`, not a vendor binary)

Artifact provenance is explicit in `backend_artifacts[*].meta`:
- IR-native artifacts: `vendor_toolchain=false`, `execution_engine=unpu_ir_runtime`
- Vendor-compiled artifacts (when `--emit-hardware-artifact` is set): `vendor_toolchain=true`

`program.json` also includes `metadata.partition_metrics`:
- `partition_count`, `cut_count`, `boundary_tensor_count`
- `ops_on_target_backend`, `ops_on_fallback_backend`
- `cost_proxy` and per-partition op counts

## Hardware-Native Artifact Emission

Enable vendor-compiler artifacts with:
- `--emit-hardware-artifact`
- `--backend-source-model <path>`

If `--emit-hardware-artifact` is set, compilation now fails if no vendor artifact is produced.

### Vela (built-in)

Requires `vela` in `PATH`.

```bash
unpu-bench ... \
  --target-format vela \
  --target-hardware hxwe2 \
  --emit-hardware-artifact \
  --backend-source-model path/to/model.tflite
```

Emits optimized `.tflite` under `out/vela/`.

### TFLM (built-in passthrough)

```bash
unpu-bench ... \
  --target-format tflm \
  --emit-hardware-artifact \
  --backend-source-model path/to/model.tflite
```

Copies source `.tflite` into `out/tflm/`.

### CVI (built-in)

Requires these binaries in `PATH`:
- `model_transform.py`
- `run_calibration.py`
- `model_deploy.py`

```bash
unpu-bench ... \
  --target-format cvi \
  --target-hardware bm1684x \
  --emit-hardware-artifact \
  --backend-source-model path/to/model.onnx \
  --data-sample path/to/sample.npy
```

Emits `.cvimodel` under `out/cvi/`.

CVI-specific flags:
- `--cvi-calibration-table`
- `--cvi-tolerance`
- `--cvi-dynamic`
- `--cvi-excepts`
- `--cvi-resize-dims`
- `--cvi-pixel-format`
- `--cvi-test-result`
- `--cvi-keep-aspect-ratio`

### eIQ (built-in)

Requires:
- `EIQ_NEUTRON_PATH=/path/to/neutron` (or `neutron` in `PATH`)

```bash
unpu-bench ... \
  --target-format eiq \
  --target-hardware mcxn947 \
  --emit-hardware-artifact \
  --backend-source-model path/to/model.tflite
```

Emits optimized `.tflite` under `out/eiq/`.

### Custom external backend command (generic fallback)

```bash
unpu-bench ... \
  --emit-hardware-artifact \
  --backend-source-model path/to/model.onnx \
  --backend-command "my_compiler --in {input} --out {out_dir}/model.bin" \
  --backend-output-glob "*.bin"
```

## Key Capabilities

- Unified IR model graph with tensor metadata, constants, and partitions
- IR canonicalization (attrs/layout normalization)
- IR structural validation pass
- Schema-driven backend legality checks (`unpu_bench/capabilities/ir_*.yaml`)
- Backend partitioning with a single contiguous accelerator core + optional fallback prefix/suffix
- E2E compile and correctness tests across multiple backends

## Correctness Guarantees and Limits

What is enforced today:
- Structural IR correctness via canonicalization + validation passes.
- Backend legality checks from explicit capability schemas.
- E2E compile tests and backend golden artifact tests.

What is not yet a formal guarantee:
- Full semantics-preservation proof from frontend graph to backend artifact.
- Tight numeric-error bounds across quantization/layout conversions for every backend.

Practical recommendation:
- Treat this as a tested engineering compiler pipeline, not a verified compiler.
- For high-stakes deployment, add task-level golden-output checks in CI using representative datasets.

## Python API

You can call conversion directly in Python:

```python
import muir

# Torch model
out = muir.convert(
    model,
    backend="vela",
    target_hardware="hxwe2",
    out_dir="out",
    input_shape=(1, 3, 32, 32),
)

# ONNX or TFLite path
out = muir.convert(
    "model.onnx",  # or "model.tflite"
    backend="cvi",
    target_hardware="bm1684x",
    out_dir="out_cvi",
)

# Cross-backend comparison report from existing program.json files
report = muir.compare_runs(
    ["out_vela/program.json", "out_cvi/program.json", "out_eiq/program.json"],
    out_dir="out_reports",
)
print(report["csv"], report["markdown"])
```

## Cross-Backend Reports

Generate automatic comparison tables from one or more `program.json` files:

```bash
unpu-bench-report \
  --program-dir out \
  --out-dir out/reports \
  --basename my_compare
```

Outputs:
- `my_compare.csv`
- `my_compare.md`

Report fields include:
- backend, topology validity, partition/cut/boundary metrics
- target/fallback op counts and fallback ratio
- IR-native vs vendor artifact counts and vendor artifact paths

## Reference Model Zoo + Random Checkpoints

Included reference model implementations used by examples:
- `dscnn` (depthwise-separable CNN)
- `mobilenet_v2`
- `resnet18`
- `convmixer`

Generate random checkpoints:

```bash
python scripts/generate_random_ckpts.py \
  --registry reference \
  --out-dir ckpts/random_reference \
  --seed 7
```

This writes `.pth` files plus `manifest.json`.

## End-to-End Python API Examples

Run all from repo root:

```bash
python examples/python_api/01_basic_torch_to_tflm.py
python examples/python_api/02_ir_edit_and_partition.py
python examples/python_api/03_torch_to_onnx_to_cvi.py
python examples/python_api/04_tflite_stub_to_eiq.py
python examples/python_api/05_batch_compile_suite.py
python examples/python_api/06_multi_backend_compare_report.py
python examples/python_api/run_all.py
```

What each example demonstrates:
- `01`: Torch model + random checkpoint -> unified IR -> TFLM backend artifacts.
- `02`: Torch -> IR canonicalize/validate -> IR edit (force one op to CPU) -> capability legality + backend-agnostic partitioning -> Vela backend artifacts.
- `03`: Torch -> ONNX export -> ONNX frontend import -> CVI backend artifacts.
- `04`: `.tflite` source -> TFLite-stub frontend -> eIQ backend artifacts.
- `05`: Batch compile across reference models/backends with one Python script.
- `06`: One reference model (`mobilenet_v2`) compiled across multiple backends, then compared with automatic CSV/Markdown report generation.
- `run_all.py`: One-shot orchestrator. Generates fake checkpoints for reference models (if missing), runs Torch/ONNX/TFLite end-to-end conversions, and writes `summary.json` + `summary.md`.

## Current Scope

- Backend artifacts are generated for multiple targets.
- Hardware-native artifact emission is first-class for `vela`, `cvi`, and `eiq`.
- `ai8x` backend remains supported via ai8x toolchain flow.
