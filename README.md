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
- `*/model.<backend>.compiled.json` (IR-native executable artifact)

## Hardware-Native Artifact Emission

Enable vendor-compiler artifacts with:
- `--emit-hardware-artifact`
- `--backend-source-model <path>`

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
- Backend partitioning with accelerator core + fallback prefix/suffix
- E2E compile and correctness tests across multiple backends

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
```

## MCU Model Zoo + Random Checkpoints

Included MCU-scale reference models:
- `dscnn_small`
- `mobilenetv2_tiny`
- `tiny_resnet8`
- `tiny_convmixer`

Generate random checkpoints:

```bash
python scripts/generate_random_ckpts.py --out-dir ckpts/random_mcu --seed 7
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
python examples/python_api/run_all.py
```

What each example demonstrates:
- `01`: Torch model + random checkpoint -> unified IR -> TFLM backend artifacts.
- `02`: Torch -> IR canonicalize/validate -> IR edit (force one op to CPU) -> capability legality + backend-agnostic partitioning -> Vela backend artifacts.
- `03`: Torch -> ONNX export -> ONNX frontend import -> CVI backend artifacts.
- `04`: `.tflite` source -> TFLite-stub frontend -> eIQ backend artifacts.
- `05`: Batch compile across models/backends with one Python script.
- `run_all.py`: One-shot orchestrator. Generates fake checkpoints for realistic MCU models (if missing), runs Torch/ONNX/TFLite end-to-end conversions, and writes `summary.json` + `summary.md`.

## Current Scope

- Backend artifacts are generated for multiple targets.
- Hardware-native artifact emission is first-class for `vela`, `cvi`, and `eiq`.
- `ai8x` backend remains supported via ai8x toolchain flow.
