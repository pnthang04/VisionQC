"""Offline tests for VisionQC pipeline helpers."""

import os
import tempfile
import unittest
from pathlib import Path

from visionqc.pipeline import checkpoint_path


class CheckpointPathTest(unittest.TestCase):
    """Tests for checkpoint discovery."""

    def test_selects_newest(self) -> None:
        """Select the newest checkpoint generated under the result directory."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / "v0/model.ckpt"
            newer = root / "v1/model.ckpt"
            older.parent.mkdir()
            newer.parent.mkdir()
            older.touch()
            newer.touch()
            older_mtime = older.stat().st_mtime - 10
            os.utime(older, (older_mtime, older_mtime))

            self.assertEqual(checkpoint_path(root, None), str(newer))

    def test_rejects_missing_file(self) -> None:
        """Reject an explicit checkpoint that does not exist."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                checkpoint_path(root, str(root / "missing.ckpt"))
