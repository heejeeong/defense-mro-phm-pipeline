"""Run one image through fine-tuned YOLO and grounded local RAG."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "rag"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "vision"))

from infer_yolo import BEST_WEIGHTS, RESULT_DIR, SAMPLE_IMAGE_DIR
from mro_rag_chain import answer_query


def run(image_path: Path, use_ollama: bool = True) -> dict:
    """Detect objects, turn detections into a query, and retrieve evidence."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install vision dependencies with: python -m pip install -r requirements.txt") from error
    if not BEST_WEIGHTS.is_file():
        raise FileNotFoundError(f"Fine-tuned weights not found: {BEST_WEIGHTS}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    model = YOLO(str(BEST_WEIGHTS))
    result = model.predict(source=str(image_path), verbose=False)[0]
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        detections.append(
            {
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": round(float(box.conf.item()), 4),
                "xyxy": [round(float(value), 1) for value in box.xyxy[0].tolist()],
            }
        )

    detection_text = ", ".join(
        f"{item['class_name']} (confidence {item['confidence']:.2f})" for item in detections
    ) or "탐지된 객체 없음"
    query = (
        f"영상 {image_path.name}에서 {detection_text} 상황이 탐지되었다. "
        "현장 확인과 대응 절차를 근거 문서와 함께 제시하라."
    )
    rag_result = answer_query(query, use_ollama=use_ollama)
    annotated_path = RESULT_DIR / f"{image_path.stem}_end_to_end{image_path.suffix}"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result.save(filename=str(annotated_path))
    return {
        "image": str(image_path.relative_to(PROJECT_ROOT)),
        "weights": str(BEST_WEIGHTS.relative_to(PROJECT_ROOT)),
        "detections": detections,
        "rag": rag_result,
        "annotated_image": str(annotated_path.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=sorted(SAMPLE_IMAGE_DIR.glob("*.jpg"))[0])
    parser.add_argument("--no-ollama", action="store_true", help="Validate detection and retrieval without generation")
    args = parser.parse_args()
    output = run(args.image, use_ollama=not args.no_ollama)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()