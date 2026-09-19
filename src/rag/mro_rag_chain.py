"""On-Premise defense RAG chain using Chroma retrieval and Ollama generation."""

import json
from pathlib import Path

try:
    from .ingest_caption import CHROMA_DIR, build_vector_store
except ImportError:
    from ingest_caption import CHROMA_DIR, build_vector_store


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = PROJECT_ROOT / "data" / "sample" / "labels"
DEFAULT_QUERY = "3번 구역 미세 이상물체 탐지 시 조치 절차는?"

def retrieve_documents(query: str, collection, top_k: int = 3) -> list[dict]:
    """Retrieve semantically similar caption chunks from Chroma."""
    result = collection.query(query_texts=[query], n_results=top_k)
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    unique = []
    seen = set()
    for text, metadata in zip(documents, metadatas):
        key = (metadata.get("source", ""), text)
        if key not in seen:
            unique.append({"text": text, "metadata": metadata})
            seen.add(key)
    return unique


def generate_with_ollama(question: str, context: list[dict], model: str = "llama3.2:3b") -> str:
    """Generate a grounded answer through the local Ollama HTTP API."""
    import json as json_module
    from urllib.error import URLError
    from urllib.request import Request, urlopen

    context_text = "\n".join(
        f"[{item['metadata'].get('source', 'unknown')}] {item['text']}" for item in context
    )
    prompt = (
        "당신은 국방 정비지원 담당자입니다. 반드시 아래 검색 근거에만 근거해 한국어로 답변하세요.\n"
        "근거에 없는 장치 설정, 센서 성능, 원인, 절차를 추정하지 마세요.\n"
        "근거에 대응 절차가 없으면 '검색 근거에 해당 절차가 없습니다'라고 답하세요.\n"
        "답변은 4개 이하의 짧은 항목으로 작성하고, 각 항목 끝에 실제 근거의 [source]를 그대로 표시하세요.\n"
        "[1], [2] 같은 숫자 각주나 별도의 출처 목록을 사용하지 말고, 검색 근거에 표시된 파일명을 직접 쓰세요.\n"
        "검색 근거에 없는 원인, 순서, 조치, 점검 항목은 추가하지 마세요.\n\n"
        f"검색 근거:\n{context_text}\n\n질의: {question}"
    )
    payload = json_module.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    ).encode("utf-8")
    request = Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            return json_module.loads(response.read().decode("utf-8"))["response"].strip()
    except (URLError, TimeoutError) as error:
        raise RuntimeError(
            "Ollama is unavailable. Install Ollama, run `ollama pull llama3.2:3b`, and retry."
        ) from error


def answer_query(question: str, collection=None, top_k: int = 3, use_ollama: bool = True) -> dict:
    """Retrieve evidence and optionally generate a local-LLM answer."""
    collection = collection or build_vector_store()
    context = retrieve_documents(question, collection, top_k=top_k)
    answer = generate_with_ollama(question, context) if use_ollama else "Ollama generation skipped."
    sources = []
    seen_sources = set()
    for item in context:
        source = item["metadata"].get("source", "unknown")
        if source not in seen_sources:
            sources.append(item["metadata"])
            seen_sources.add(source)
    return {"question": question, "answer": answer, "sources": sources}


def main() -> None:
    """Run a local operational guidance query without external APIs."""
    result = answer_query(DEFAULT_QUERY)
    print(f"Chroma path: {CHROMA_DIR}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
