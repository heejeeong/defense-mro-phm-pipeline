# defense-mro-phm-pipeline

> **국방 멀티모달 데이터 기반 무기체계 이상탐지(YOLO) 및 정비지원(RAG) AI 파이프라인**

## Project Overview

한화에어로스페이스 MRO(무기체계 운용/정비지원 S/W 플랫폼) 직무 적용을 목표로 구축한 미니 프로젝트입니다.
국방 경계 작전 환경 데이터셋을 활용하여 비전 기반 이상 징후 탐지(PHM)와 폐쇄망 환경을 고려한 로컬 sLLM 기반 정비/상황 조치 지원(RAG)을 통합 검증합니다.

## Key Architecture

1. **Vision Engine (YOLO)**: 군 기지 및 무기체계 주변 이상 객체/파손 징후 실시간 Bounding Box 탐지
2. **MRO RAG Engine (LangChain + sLLM)**: 감시 캡션 및 무기체계 정비 지침 데이터를 Vector DB화하여 이상 탐지 발생 시 정비사 대응 절차 자동 추천

## Dataset

- **AI Hub**: 군 경계 작전 환경 내 인식 데이터 (이미지 + Bounding Box + 상황 캡션 텍스트)

## Future Enhancements

- 소형 객체 탐지 정밀도 향상을 위한 SAHI 적용
- On-Premise 경량화 sLLM(Qwen2.5 / Llama-3) 튜닝

## Project Structure

```text
data/raw_sample/       # 국방 데이터 샘플, 원본 데이터는 저장하지 않음
src/vision/            # YOLO 학습 및 추론
src/rag/               # 캡션 임베딩 및 MRO RAG 체인
docs/result_images/    # 탐지 결과 이미지
```
