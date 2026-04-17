"""Regenerate rfs.pb.h / rfs.pb.c (nanopb, for the Teensy firmware) and
rfs_pb2.py (standard protobuf, for the Python host) from rfs.proto.

Run from anywhere:
    python protocol/regen.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nanopb  # type: ignore

HERE = Path(__file__).resolve().parent
NANOPB_DIR = Path(nanopb.__file__).parent / "generator"


def main() -> None:
    # 1. nanopb C sources for the firmware
    subprocess.run(
        [sys.executable, str(NANOPB_DIR / "nanopb_generator.py"), "rfs.proto"],
        cwd=str(HERE), check=True,
    )

    # 2. Standard protobuf Python module for the host
    subprocess.run(
        [sys.executable, "-m", "grpc_tools.protoc",
         "--proto_path=.",
         f"--proto_path={NANOPB_DIR / 'proto'}",
         "--python_out=.",
         "rfs.proto"],
        cwd=str(HERE), check=True,
    )

    print("Generated: rfs.pb.h, rfs.pb.c, rfs_pb2.py")


if __name__ == "__main__":
    main()
