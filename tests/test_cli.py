import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "lucas_moll", *arguments],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_import_performs_no_simulation_or_filesystem_writes(self):
        root = str(Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"import sys; sys.path.insert(0, {root!r}); import lucas_moll",
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_config_override_saved_outputs_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"particles": 20, "final_time": 0.004, "snapshots": 3}))
            output = root / "run"
            args = (
                "--config",
                str(config),
                "--particles",
                "30",
                "--output",
                str(output),
                "--no-plots",
                "--quiet",
            )
            completed = self.run_cli(*args)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["config"]["particles"], 30)
            with np.load(output / "snapshots.npz", allow_pickle=False) as data:
                self.assertEqual(data["pde_density"].shape, (3, 321))
                self.assertEqual(data["final_particles"].shape, (30,))
            before = {file.name: file.read_bytes() for file in output.iterdir()}
            completed = self.run_cli(*args)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("empty directory", completed.stderr)
            self.assertEqual(before, {file.name: file.read_bytes() for file in output.iterdir()})

    def test_invalid_configuration_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            completed = self.run_cli("--dt", "10", "--output", str(output))
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
