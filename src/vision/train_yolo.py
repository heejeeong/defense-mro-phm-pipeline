"""Train and validate a lightweight YOLOv8 defense anomaly detector."""

import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG = PROJECT_ROOT / "data" / "sample" / "yolo" / "dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "train"
RUN_NAME = "defense_anomaly_exp"

logger = logging.getLogger(__name__)


def train_yolo() -> None:
    """Fine-tune YOLOv8n and report final validation mAP metrics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("The 'ultralytics' package is not installed.")
        logger.error("Install it with: python -m pip install ultralytics")
        return

    if not DATASET_CONFIG.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {DATASET_CONFIG}")

    logger.info("Loading pretrained model: yolov8n.pt")
    model = YOLO("yolov8n.pt")

    logger.info("Starting YOLOv8 training for 10 epochs.")
    model.train(
        data=str(DATASET_CONFIG),
        epochs=10,
        imgsz=640,
        batch=8,
        project=str(RUNS_DIR),
        name=RUN_NAME,
    )
    logger.info("Training completed. Running validation.")

    metrics = model.val(
        data=str(DATASET_CONFIG),
        imgsz=640,
        batch=8,
        project=str(RUNS_DIR),
        name=f"{RUN_NAME}_val",
    )
    logger.info("Final validation metrics:")
    logger.info("mAP50: %.4f", metrics.box.map50)
    logger.info("mAP50-95: %.4f", metrics.box.map)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_yolo()

