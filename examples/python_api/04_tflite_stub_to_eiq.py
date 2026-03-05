from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import muir


def main() -> None:
    out_dir = Path("out/examples/04_tflite_to_eiq")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Demo uses a placeholder .tflite so you can exercise IR + lowering flow without vendor tools.
    tflite_path = out_dir / "placeholder.tflite"
    tflite_path.write_bytes(b"FAKE_TFLITE")

    result = muir.convert(
        str(tflite_path),
        backend="eiq",
        target_hardware="mcxn947",
        out_dir=str(out_dir),
        bit_width=8,
    )
    print("Artifacts:", [a["path"] for a in result["artifacts"]])
    print("Program:", out_dir / "program.json")


if __name__ == "__main__":
    main()
