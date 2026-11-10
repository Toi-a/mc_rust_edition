"""The Python side of the wire format agrees with crates/horizon-rl-proto (the same hand-written
bytes as its `step_bytes_are_little_endian_in_field_order` test)."""
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horizon_rl import proto  # noqa: E402


class ProtoTest(unittest.TestCase):
    def test_step_matches_the_rust_layout(self):
        payload = bytes([0x02, 0x01, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0x81, 0, 0, 0, 2, 0, 0, 0,
                         0x00, 0x00, 0xb4, 0x42, 0x00, 0x00, 0x34, 0xc2])
        msg = proto.step(0x0102, proto.FORWARD | proto.ATTACK, 2, 90.0, -45.0, frame_skip=3)
        self.assertEqual(msg, struct.pack("<II", proto.STEP, 28) + payload)

    def test_buttons_from_names(self):
        self.assertEqual(proto.buttons_of({"forward": True, "use": 1, "jump": False}),
                         proto.FORWARD | proto.USE)

    def test_header_reads_back(self):
        h = b"HZRL" + struct.pack("<IQQIIIIIII", 1, 7, 99, 64, 70, 344, 19, 448, 0, 0)
        got = proto.read_header(h)
        self.assertEqual((got["magic"], got["seq"], got["tick"], got["state_len"]), (b"HZRL", 7, 99, 70))


if __name__ == "__main__":
    unittest.main()
