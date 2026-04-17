"""Reusable USB-serial link to the Teensy RFS firmware.

Wire framing: [u8 type][u16 seq][u16 len][protobuf payload], little-endian.
Import this from any test script:

    from rfs_link import Link, pb, PING, PONG, CONFIG, STEP, TRACKS

    link = Link("COM7")
    link.send(PING); link.recv(link.seq, PONG)
    link.send(CONFIG, pb.Config(state_dim=4, meas_dim=2, ...))
    seq = link.send(STEP, pb.Step(timestep=0))
    tracks = pb.Tracks(); tracks.ParseFromString(link.recv(seq, TRACKS))
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import serial

# rfs_pb2 lives in ../protocol/ and imports nanopb_pb2 from inside the nanopb
# pip package. Make both findable.
import nanopb  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "protocol"))
sys.path.insert(0, os.path.join(os.path.dirname(nanopb.__file__), "generator", "proto"))
import rfs_pb2 as pb  # type: ignore  # noqa: E402

_HDR = "<B H H"

PING, PONG, RESET, CONFIG, STEP, TRACKS, LOG, ERROR = 1, 2, 3, 4, 5, 6, 7, 0x7F


class Link:
    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 30.0):
        self.ser = serial.Serial(port, baud, timeout=timeout_s)
        self.seq = 0

    def close(self) -> None:
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def send(self, mtype: int, msg=None) -> int:
        self.seq = (self.seq + 1) & 0xFFFF
        payload = msg.SerializeToString() if msg is not None else b""
        self.ser.write(struct.pack(_HDR, mtype, self.seq, len(payload)) + payload)
        return self.seq

    def recv(self, want_seq: int, want_type: int) -> bytes:
        """Read frames until one matches (want_seq, want_type). LOGs print, ERRORs raise."""
        while True:
            h = self.ser.read(5)
            if len(h) != 5:
                raise TimeoutError(f"short header read: got {len(h)} bytes")
            mtype, seq, plen = struct.unpack(_HDR, h)
            payload = self.ser.read(plen) if plen else b""
            if mtype == LOG:
                m = pb.Log(); m.ParseFromString(payload)
                print(f"[mcu log] {m.msg}")
                continue
            if mtype == ERROR:
                m = pb.Error(); m.ParseFromString(payload)
                raise RuntimeError(f"mcu error: {m.msg}")
            if seq == want_seq and mtype == want_type:
                return payload
