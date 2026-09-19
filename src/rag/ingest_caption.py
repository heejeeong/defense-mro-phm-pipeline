"""Ingest surveillance captions into a persistent Chroma collection."""

import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = PROJECT_ROOT / "data" / "sample" / "labels"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "defense_captions"
GUIDANCE_RECORDS = [
    {
        "text": "이상 객체가 탐지되면 먼저 해당 영상과 탐지 시각을 고정하고 동일 위치를 재탐지한다.",
        "source": "docs/response_guidance.md",
        "image": "",
        "zone": "all",
    },
    {
        "text": "동일 위치에서 반복 탐지되면 관제 담당자에게 위치와 탐지 결과를 보고하고 인접 센서 상태를 확인한다.",
        "source": "docs/response_guidance.md",
        "image": "",
        "zone": "all",
    },
    {
        "text": "반복 탐지가 확인되면 정비 담당자가 현장 외관과 체결 상태를 점검하고 조치 결과를 운용 로그에 기록한다.",
        "source": "docs/response_guidance.md",
        "image": "",
        "zone": "all",
    },
]


def load_caption_records() -> list[dict[str, str]]:
    """Read caption annotations and retain their source metadata."""
    records = []
    for json_path in sorted(LABEL_DIR.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for annotation in payload.get("annotations", []):
            caption = str(annotation.get("caption", "")).strip()
            if caption:
                records.append(
                    {
                        "text": caption,
                        "source": json_path.name,
                        "image": str(annotation.get("filename", "")),
                        "zone": str(payload.get("meta", {}).get("spot", "unknown")),
                    }
                )
    return records + GUIDANCE_RECORDS


def build_vector_store(reset: bool = False):
    """Create or update the local Chroma collection using real embeddings."""
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError as error:
        raise RuntimeError(
            "Install RAG dependencies with: python -m pip install -r requirements.txt"
        ) from error

    records = load_caption_records()
    if not records:
        raise FileNotFoundError(f"No caption records found in {LABEL_DIR}")
    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    existing_ids = set(collection.get(include=[]).get("ids", []))
    pending = [
        (f"caption-{index}", record)
        for index, record in enumerate(records)
        if f"caption-{index}" not in existing_ids
    ]
    if pending:
        collection.add(
            ids=[item[0] for item in pending],
            documents=[item[1]["text"] for item in pending],
            metadatas=[
                {key: value for key, value in item[1].items() if key != "text"}
                for item in pending
            ],
        )
    return collection


def main() -> None:
    """Build the vector store and report its document count."""
    collection = build_vector_store(reset=True)
    print(f"Indexed {collection.count()} caption chunks in {CHROMA_DIR}.")


if __name__ == "__main__":
    main()
