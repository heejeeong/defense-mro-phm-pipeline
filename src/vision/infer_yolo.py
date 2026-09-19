"""Run YOLO inference on sample defense images and save annotated results."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BEST_WEIGHTS = PROJECT_ROOT / "runs" / "train" / "defense_anomaly_exp-2" / "weights" / "best.pt"
SAMPLE_IMAGE_DIR = PROJECT_ROOT / "data" / "sample" / "images"
RESULT_DIR = PROJECT_ROOT / "docs" / "result_images"
SAMPLE_COUNT = 5


def infer_yolo() -> list[Path]:
    """Run inference on up to three sample images and return result paths."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "The 'ultralytics' package is not installed. "
            "Install it with: python -m pip install ultralytics"
        ) from error

    image_paths = sorted(
        path
        for path in SAMPLE_IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:SAMPLE_COUNT]
    if not image_paths:
        raise FileNotFoundError(f"No sample images found in {SAMPLE_IMAGE_DIR}")

    if not BEST_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Fine-tuned weights not found: {BEST_WEIGHTS}. Run src/vision/train_yolo.py first."
        )
    weights = BEST_WEIGHTS
    print(f"Using weights: {weights}")
    model = YOLO(str(weights))
    results = model.predict(source=[str(path) for path in image_paths], verbose=False)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for image_path, result in zip(image_paths, results):
        output_path = RESULT_DIR / f"{image_path.stem}_detected{image_path.suffix}"
        result.save(filename=str(output_path))
        output_paths.append(output_path)

    print("Inference complete. Output files:")
    for output_path in output_paths:
        print(output_path)
    return output_paths


if __name__ == "__main__":
    infer_yolo()
