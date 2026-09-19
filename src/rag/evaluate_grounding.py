"""Run a reproducible ten-question grounded-answer evaluation."""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "rag"))

from mro_rag_chain import answer_query


CASES = [
    {"id": "Q01", "question": "이상 객체가 처음 탐지되면 가장 먼저 무엇을 해야 하나요?", "evidence_terms": ["탐지 시각", "동일 위치", "재탐지"]},
    {"id": "Q02", "question": "미세 이상물체 탐지 직후 영상 확인 절차를 알려주세요.", "evidence_terms": ["영상", "탐지 시각", "고정"]},
    {"id": "Q03", "question": "첫 탐지의 오탐 여부를 확인하기 위한 근거 문서 절차는 무엇인가요?", "evidence_terms": ["동일 위치", "재탐지"]},
    {"id": "Q04", "question": "같은 위치에서 이상 객체가 반복되면 관제 담당자에게 무엇을 보고하나요?", "evidence_terms": ["위치", "탐지 결과", "보고"]},
    {"id": "Q05", "question": "반복 탐지 시 인접 설비와 관련해 확인할 항목은 무엇인가요?", "evidence_terms": ["인접 센서", "상태", "확인"]},
    {"id": "Q06", "question": "반복 이상이 확인되었을 때 관제 보고와 센서 점검 순서를 알려주세요.", "evidence_terms": ["관제", "인접 센서", "보고"]},
    {"id": "Q07", "question": "반복 탐지 후 정비 담당자는 현장에서 무엇을 점검해야 하나요?", "evidence_terms": ["외관", "체결 상태", "점검"]},
    {"id": "Q08", "question": "정비 담당자의 반복 이상 확인 후 현장 점검 범위를 알려주세요.", "evidence_terms": ["현장", "외관", "체결 상태"]},
    {"id": "Q09", "question": "탐지 대응 결과는 어디에 기록해야 하나요?", "evidence_terms": ["조치 결과", "운용 로그", "기록"]},
    {"id": "Q10", "question": "이상 객체 대응을 완료한 뒤 기록할 내용과 기록 위치를 알려주세요.", "evidence_terms": ["탐지", "조치 결과", "운용 로그"]},
]


def evaluate_case(case: dict, result: dict) -> dict:
    answer = result["answer"]
    retrieved_sources = {metadata.get("source", "unknown") for metadata in result.get("sources", [])}
    cited_sources = set(re.findall(r"\[([^\]]+)\]", answer))
    source_present = bool(retrieved_sources) and any(source in answer for source in retrieved_sources)
    citations_are_retrieved = cited_sources.issubset(retrieved_sources)
    matched_terms = [term for term in case["evidence_terms"] if term in answer]
    evidence_supported = len(matched_terms) >= 2
    failures = []
    if not source_present:
        failures.append("retrieved source is not shown in the answer")
    if not citations_are_retrieved:
        failures.append("answer cites a source that was not retrieved")
    if not evidence_supported:
        failures.append("fewer than two expected evidence terms are present")
    return {
        "id": case["id"],
        "question": case["question"],
        "answer": answer,
        "retrieved_sources": sorted(retrieved_sources),
        "cited_sources": sorted(cited_sources),
        "matched_evidence_terms": matched_terms,
        "passed": source_present and citations_are_retrieved and evidence_supported,
        "failures": failures,
    }


def build_report(use_ollama: bool) -> dict:
    results = []
    for case in CASES:
        result = answer_query(case["question"], use_ollama=use_ollama)
        results.append(evaluate_case(case, result))
    passed = sum(item["passed"] for item in results)
    source_compliant = sum(
        bool(item["retrieved_sources"])
        and bool(set(item["cited_sources"]).issubset(item["retrieved_sources"]))
        and any(source in item["answer"] for source in item["retrieved_sources"])
        for item in results
    )
    evidence_supported = sum(len(item["matched_evidence_terms"]) >= 2 for item in results)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "ollama_generation" if use_ollama else "retrieval_only",
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 3),
        "source_compliant": source_compliant,
        "source_compliance_rate": round(source_compliant / len(results), 3),
        "evidence_supported": evidence_supported,
        "evidence_support_rate": round(evidence_supported / len(results), 3),
        "results": results,
    }


def write_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    lines = [
        "# E2E Grounding Evaluation", "",
        f"- Mode: `{report['mode']}`",
        f"- Result: **{report['passed']}/{report['total']} passed** ({report['pass_rate']:.1%})",
        f"- Source compliance: **{report['source_compliant']}/{report['total']}**",
        f"- Evidence-term support: **{report['evidence_supported']}/{report['total']}**", "",
        "The pass rule requires a retrieved source to appear in the answer, every bracketed citation to be retrieved, and at least two expected evidence terms to appear.",
        "This is a reproducible lexical check, not a complete semantic proof that no unsupported claim was generated.", "",
        "| ID | Pass | Sources | Evidence terms | Failure |", "| --- | --- | --- | --- | --- |",
    ]
    for item in report["results"]:
        lines.append(
            f"| {item['id']} | {'yes' if item['passed'] else 'no'} | "
            f"{', '.join(item['retrieved_sources'])} | {len(item['matched_evidence_terms'])} | "
            f"{'; '.join(item['failures'])} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-ollama", action="store_true", help="Check retrieval only; do not count this as generated-answer validation.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs" / "e2e_grounding_report.json")
    args = parser.parse_args()
    report = build_report(use_ollama=not args.no_ollama)
    write_report(report, args.output)
    print(json.dumps({key: report[key] for key in report if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()