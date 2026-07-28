"""Offline tests for VisionQC pipeline helpers."""

import os
import tempfile
import unittest
from pathlib import Path

from lightning.pytorch.callbacks import EarlyStopping

from visionqc.pipeline import BatchedEfficientAd, DEFAULT_CONFIG, checkpoint_path, load_config, make_engine, make_model


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


class ValidationAndEarlyStoppingTest(unittest.TestCase):
    """Tests for validation metrics and training callbacks."""

    def test_validation_metric_is_configured(self) -> None:
        """Expose image AUROC to checkpointing and early stopping."""
        model = make_model(load_config(DEFAULT_CONFIG))

        self.assertEqual([metric.name for metric in model.evaluator.val_metrics], ["image_AUROC"])

    def test_batched_multi_gpu_training_is_configured(self) -> None:
        """Use batch 32 per device across both available GPUs."""
        config = load_config(DEFAULT_CONFIG)
        model = make_model(config)

        self.assertIsInstance(model, BatchedEfficientAd)
        self.assertNotIn("evaluator", model.hparams)
        self.assertNotIn("visualizer", model.hparams)
        self.assertEqual(config.dataset.train_batch_size, 32)
        self.assertEqual(config.trainer.devices, 2)
        self.assertEqual(config.trainer.strategy, "ddp_find_unused_parameters_true")
        self.assertEqual(config.trainer.precision, "16-mixed")
        self.assertEqual(config.trainer.max_steps, 1100)

    def test_validation_is_disjoint_from_test(self) -> None:
        """Split validation from test instead of evaluating on the same samples."""
        config = load_config(DEFAULT_CONFIG)

        self.assertEqual(config.dataset.val_split_mode, "from_test")
        self.assertEqual(config.dataset.val_split_ratio, 0.5)

    def test_early_stopping_is_enabled_for_full_training(self) -> None:
        """Monitor validation AUROC during a full training run."""
        config = load_config(DEFAULT_CONFIG)
        engine = make_engine(config)

        callbacks = engine._cache.args["callbacks"]  # noqa: SLF001
        early_stopping = next(callback for callback in callbacks if isinstance(callback, EarlyStopping))
        self.assertEqual(early_stopping.monitor, "image_AUROC")
