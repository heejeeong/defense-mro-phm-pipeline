"""On-Premise defense RAG demonstration using local sample captions."""

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = PROJECT_ROOT / "data" / "sample" / "labels"
DEFAULT_QUERY = "3번 구역 미세 이상물체 탐지 시 조치 절차는?"

FALLBACK_LOGS = [
    "3번 구역 감시 영상에서 미세 이상물체가 탐지되었다. 해당 구역의 영상을 고정하고 재탐지한다.",
    "이상 징후 발생 시 현장 접근 전 관제 담당자에게 상황을 보고하고 인접 센서의 상태를 확인한다.",
    "동일 위치에서 이상물체가 반복 탐지되면 정비 담당자가 외관과 체결 상태를 점검하고 조치 결과를 기록한다.",
    "오탐으로 판단되더라도 탐지 시각, 구역, 객체 유형, 조치 결과를 운용 로그에 남긴다.",
]


def load_documents():
    """Load caption documents from sample JSON labels, with an offline fallback."""
    try:
        from langchain_core.documents import Document
    except ImportError as error:
        raise RuntimeError(
            "LangChain is required. Install it with: python -m pip install langchain"
        ) from error

    documents = []
    for json_path in sorted(LABEL_DIR.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for annotation in payload.get("annotations", []):
            caption = str(annotation.get("caption", "")).strip()
            if caption:
                documents.append(
                    Document(
                        page_content=caption,
                        metadata={"source": json_path.name, "zone": payload.get("meta", {}).get("spot")},
                    )
                )

    if documents:
        return documents
    return [Document(page_content=text, metadata={"source": "offline_fallback"}) for text in FALLBACK_LOGS]


def retrieve_documents(query: str, documents: list, top_k: int = 3) -> list:
    """Retrieve documents using a deterministic local token-overlap scorer."""
    tokens = set(re.findall(r"[0-9A-Za-z가-힣]+", query.lower()))
    ranked = sorted(
        documents,
        key=lambda document: len(tokens & set(re.findall(r"[0-9A-Za-z가-힣]+", document.page_content.lower()))),
        reverse=True,
    )
    return ranked[:top_k]


def build_qa_chain(documents: list):
    """Build a LangChain Runnable chain with a local retriever and answer function."""
    try:
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    except ImportError as error:
        raise RuntimeError(
            "LangChain is required. Install it with: python -m pip install langchain"
        ) from error

    retriever = RunnableLambda(lambda query: retrieve_documents(query, documents))

    def answer(inputs: dict) -> str:
        context = inputs["context"]
        context_text = "\n".join(f"- {document.page_content}" for document in context)
        return (
            "[On-Premise Defense AI Assistant]\n"
            f"질의: {inputs['question']}\n\n"
            "검색된 현장 근거:\n"
            f"{context_text}\n\n"
            "권고 조치:\n"
            "1. 해당 구역의 영상을 고정하고 동일 위치를 재탐지합니다.\n"
            "2. 관제 담당자에게 탐지 시각과 위치를 보고하고 인접 센서 상태를 확인합니다.\n"
            "3. 반복 탐지 시 정비 담당자가 현장 외관과 체결 상태를 점검합니다.\n"
            "4. 탐지 및 조치 결과를 운용 로그에 기록합니다."
        )

    return {"context": retriever, "question": RunnablePassthrough()} | RunnableLambda(answer)


def main() -> None:
    """Run a local operational guidance query without external APIs."""
    documents = load_documents()
    qa_chain = build_qa_chain(documents)
    print(f"Loaded {len(documents)} local defense documents.")
    print(qa_chain.invoke(DEFAULT_QUERY))


if __name__ == "__main__":
    main()
