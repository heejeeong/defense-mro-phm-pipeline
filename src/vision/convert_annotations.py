"""Convert AI Hub JSON bounding boxes into YOLO label files."""

import json
import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
JSON_LABEL_DIR = SAMPLE_DIR / "labels"
YOLO_LABEL_DIR = JSON_LABEL_DIR
DATASET_YAML = SAMPLE_DIR / "yolo" / "dataset.yaml"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure readable conversion logs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def normalize_bbox(bbox: list[float], width: int, height: int) -> tuple[float, ...]:
    """Convert [x, y, box_width, box_height] to normalized YOLO coordinates."""
    if len(bbox) != 4 or width <= 0 or height <= 0:
        raise ValueError("Invalid bounding box or image dimensions")

    x, y, box_width, box_height = bbox
    center_x = (x + box_width / 2) / width
    center_y = (y + box_height / 2) / height
    normalized_width = box_width / width
    normalized_height = box_height / height
    return center_x, center_y, normalized_width, normalized_height


def convert_label(json_path: Path, output_path: Path) -> int:
    """Convert one AI Hub JSON file and return its annotation count."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for annotation in payload.get("annotations", []):
        width = int(annotation["width"])
        height = int(annotation["height"])
        values = normalize_bbox(annotation["bbox"], width, height)
        class_id = int(annotation["class"])
        lines.append(f"{class_id} {' '.join(f'{value:.6f}' for value in values)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_dataset_yaml(class_ids: set[int]) -> None:
    """Write a minimal Ultralytics dataset configuration."""
    DATASET_YAML.parent.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {class_id}: class_{class_id}" for class_id in sorted(class_ids))
    DATASET_YAML.write_text(
        f"path: {SAMPLE_DIR.as_posix()}\n"
        "train: images\n"
        "val: images\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )


def convert_dataset() -> None:
    """Convert all sample JSON labels into YOLO text labels."""
    json_files = sorted(JSON_LABEL_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON labels found in {JSON_LABEL_DIR}")

    converted_annotations = 0
    class_ids: set[int] = set()
    for json_path in json_files:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        class_ids.update(int(annotation["class"]) for annotation in payload.get("annotations", []))
        output_path = YOLO_LABEL_DIR / f"{json_path.stem}.txt"
        converted_annotations += convert_label(json_path, output_path)

    write_dataset_yaml(class_ids)
    logger.info("Converted %d JSON files and %d annotations.", len(json_files), converted_annotations)
    logger.info("YOLO labels: %s", YOLO_LABEL_DIR)
    logger.info("Dataset config: %s", DATASET_YAML)


if __name__ == "__main__":
    configure_logging()
    convert_dataset()