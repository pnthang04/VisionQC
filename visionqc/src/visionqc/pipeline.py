"""VisionQC EfficientAD baseline on VisA/pcb1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from omegaconf import OmegaConf

from anomalib.callbacks import ModelCheckpoint
from anomalib.data import Visa
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, Evaluator, F1Score
from anomalib.models import EfficientAd
from anomalib.visualization import ImageVisualizer

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "visionqc/configs/efficientad_pcb1.yaml"


class BatchedEfficientAd(EfficientAd):
    """Allow batched project training without modifying Anomalib core."""

    def on_train_start(self) -> None:
        """Run upstream initialization while retaining the configured train loader."""
        datamodule = self.trainer.datamodule
        configured_batch_size = datamodule.train_batch_size
        datamodule.train_batch_size = 1
        try:
            super().on_train_start()
        finally:
            datamodule.train_batch_size = configured_batch_size


def load_config(path: Path) -> Any:
    """Load the baseline configuration and resolve paths from the repository root."""
    config = OmegaConf.load(path)
    config.dataset.root = str(ROOT / config.dataset.root)
    config.model.imagenet_dir = str(ROOT / config.model.imagenet_dir)
    config.output_dir = str(ROOT / config.output_dir)
    return config


def make_datamodule(config: Any) -> Visa:
    """Create the official VisA datamodule."""
    return Visa(**OmegaConf.to_container(config.dataset, resolve=True))


def make_model(config: Any) -> EfficientAd:
    """Create EfficientAD with detection, localization, F1, and AU-PRO metrics."""
    val_metrics = [
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
    ]
    metrics = [
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
        F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
        AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
        AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
    ]
    output_dir = Path(config.output_dir)
    model = BatchedEfficientAd(
        **OmegaConf.to_container(config.model, resolve=True),
        evaluator=Evaluator(val_metrics=val_metrics, test_metrics=metrics),
        visualizer=ImageVisualizer(output_dir=output_dir / "heatmaps"),
    )
    # These modules are already part of state_dict. Keeping their live objects
    # in hyperparameters makes DDP checkpoints try to pickle process locks.
    model.hparams.pop("evaluator", None)
    model.hparams.pop("visualizer", None)
    return model


def make_engine(config: Any, smoke: bool = False) -> Engine:
    """Create the Anomalib engine."""
    trainer = OmegaConf.to_container(config.trainer, resolve=True)
    callbacks = [
        ModelCheckpoint(
            filename="model-best",
            monitor=config.early_stopping.monitor,
            mode=config.early_stopping.mode,
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]
    if config.early_stopping.enabled and not smoke:
        callbacks.append(
            EarlyStopping(
                monitor=config.early_stopping.monitor,
                mode=config.early_stopping.mode,
                patience=config.early_stopping.patience,
                min_delta=config.early_stopping.min_delta,
                check_finite=True,
                verbose=True,
            ),
        )
    if smoke:
        trainer.update(max_epochs=1, max_steps=2, limit_train_batches=2, limit_val_batches=2, limit_test_batches=2)
    return Engine(default_root_dir=config.output_dir, logger=True, callbacks=callbacks, **trainer)


def prepare(config: Any) -> Visa:
    """Download, convert, inspect, and report VisA/pcb1."""
    datamodule = make_datamodule(config)
    datamodule.prepare_data()
    datamodule.setup()
    train = datamodule.train_data.samples
    val = datamodule.val_data.samples
    test = datamodule.test_data.samples
    batch = next(iter(datamodule.train_dataloader()))
    report = {
        "path": str(datamodule.split_root / datamodule.category),
        "train_normal": int((train.label_index == 0).sum()),
        "validation": len(val),
        "test_normal": int((test.label_index == 0).sum()),
        "test_anomalous": int((test.label_index == 1).sum()),
        "masks": int(test.mask_path.notna().sum()),
        "batch_shape": list(batch.image.shape),
    }
    print(json.dumps(report, indent=2))
    return datamodule


def json_value(value: Any) -> Any:
    """Convert tensors and paths into JSON-compatible values."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_metrics(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Save test metrics as JSON and CSV."""
    rows = [{key: json_value(value) for key, value in row.items()} for row in results]
    (output_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


def save_predictions(predictions: list[Any], output_dir: Path) -> None:
    """Save anomaly scores and representative TN/TP/FP/FN visualizations."""
    rows: list[dict[str, Any]] = []
    examples: dict[str, Any] = {}
    for batch in predictions:
        for item in batch:
            gt = int(json_value(item.gt_label))
            pred = int(json_value(item.pred_label))
            outcome = ("t" if gt == pred else "f") + ("p" if pred else "n")
            rows.append(
                {
                    "image_path": str(item.image_path),
                    "gt_label": gt,
                    "pred_label": pred,
                    "anomaly_score": float(json_value(item.pred_score)),
                    "outcome": outcome,
                },
            )
            examples.setdefault(outcome, item)

    with (output_dir / "anomaly_scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "gt_label", "pred_label", "anomaly_score", "outcome"])
        writer.writeheader()
        writer.writerows(rows)

    visualizer = ImageVisualizer()
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    for outcome, item in examples.items():
        image = visualizer(item)
        if image is not None:
            image.save(examples_dir / f"{outcome}.png")


def checkpoint_path(output_dir: Path, requested: str | None) -> str:
    """Resolve an explicit checkpoint or the newest generated checkpoint."""
    if requested:
        path = Path(requested)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            return str(path)
        raise FileNotFoundError(path)
    checkpoints = sorted(output_dir.rglob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under {output_dir}")
    return str(checkpoints[0])


def run_train(config: Any, resume: str | None, smoke: bool = False) -> None:
    """Train and test EfficientAD."""
    datamodule = prepare(config)
    model = make_model(config)
    engine = make_engine(config, smoke=smoke)
    resume_path = checkpoint_path(Path(config.output_dir), resume) if resume else None
    engine.fit(model=model, datamodule=datamodule, ckpt_path=resume_path)
    best = engine.best_model_path or resume_path
    results = engine.test(model=model, datamodule=datamodule, ckpt_path=best)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(results, output_dir)
    (output_dir / "best_checkpoint.txt").write_text(str(best), encoding="utf-8")
    if smoke:
        assert best and Path(best).is_file(), "Smoke test did not create a checkpoint"
        print(f"Smoke test passed: {best}")


def evaluate(config: Any, checkpoint: str | None) -> None:
    """Evaluate a checkpoint and persist metrics, scores, maps, and examples."""
    datamodule = prepare(config)
    model = make_model(config)
    engine = make_engine(config)
    output_dir = Path(config.output_dir)
    ckpt = checkpoint_path(output_dir, checkpoint)
    results = engine.test(model=model, datamodule=datamodule, ckpt_path=ckpt)
    predictions = engine.predict(model=model, datamodule=datamodule, ckpt_path=ckpt, return_predictions=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(results, output_dir)
    save_predictions(predictions, output_dir)
    print(f"Evaluated checkpoint: {ckpt}")


def main() -> None:
    """Run a VisionQC pipeline command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "smoke", "train", "evaluate"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config.seed, workers=True)
    if args.command == "prepare":
        prepare(config)
    elif args.command == "smoke":
        run_train(config, resume=None, smoke=True)
    elif args.command == "train":
        run_train(config, resume=args.checkpoint)
    else:
        evaluate(config, args.checkpoint)
