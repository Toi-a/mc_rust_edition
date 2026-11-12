"""FleetConfig: names, opponents and config files (no game needed)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horizon_rl.fleet import FleetConfig  # noqa: E402


class FleetConfigTest(unittest.TestCase):
    def test_pairs_face_each_other_and_an_odd_one_out_has_none(self):
        c = FleetConfig(server="h:1", n=5, names="Bot{i}", pairs=True)
        self.assertEqual([c.target_of(i) for i in range(5)], ["Bot1", "Bot0", "Bot3", "Bot2", None])

    def test_a_target_pattern_names_each_opponent(self):
        c = FleetConfig(server="h:1", n=2, target="Rival_{i}")
        self.assertEqual(c.target_of(1), "Rival_1")
        self.assertIsNone(FleetConfig(server="h:1").target_of(0), "nearest player by default")

    def test_a_toml_file_with_overrides_on_top(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fleet.toml"
            path.write_text('server = "10.0.0.2:25565"\nn = 6\nextra_args = ["--foo"]\n')
            c = FleetConfig.load(path, n=3, render_distance=None)
            self.assertEqual((c.server, c.n, c.render_distance, c.extra_args),
                             ("10.0.0.2:25565", 3, 2, ("--foo",)))
            path.write_text('server = "h:1"\nnn = 6\n')
            with self.assertRaisesRegex(ValueError, "nn"):
                FleetConfig.load(path)

    def test_the_example_config_loads(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "fleet.toml"
        self.assertTrue(FleetConfig.load(example).pairs)


if __name__ == "__main__":
    unittest.main()
